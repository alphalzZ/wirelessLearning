
# transformer_ofdm_schemeC_plus_demo.py
# -*- coding: utf-8 -*-
"""
Upgraded Scheme-C (RE-level tokens) demo for OFDM LLR prediction
- Conv-Stem: light 2D conv over (n_sym, rb_sc) grid by scatter -> conv2d -> gather
- Neighborhood-sparse cross-attention: only attend to pilot REs within |Δk| <= K
- Parallel analytic LLR head (max-log QAM demapper) fused with learned head
This is a training scaffold using dummy inputs/labels (no real OFDM modeling).
"""
import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Helpers: token (k, n) positions for RE-level tokens
# ----------------------------
def build_re_positions(rb_sc: int = 12, n_sym: int = 14,
                       pilot_syms=(2, 10)) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return encoder and decoder positions as integer indices (k, n)."""
    pilot_set = set(pilot_syms)
    enc, dec = [], []
    for n in range(n_sym):
        for k in range(rb_sc):
            if n in pilot_set and k%2==0:
                enc.append((k, n))
            elif n not in pilot_set:
                dec.append((k, n))
    enc_pos = torch.tensor(enc, dtype=torch.long)  # [24, 2]
    dec_pos = torch.tensor(dec, dtype=torch.long)  # [144, 2]
    #print(enc_pos.shape)
    #print(dec_pos.shape)
    assert enc_pos.shape[0] == 12 and dec_pos.shape[0] == 144
    return enc_pos, dec_pos

def build_re_positions_3rb(
    rb_sc: int = 12,
    n_sym: int = 14,
    pilot_syms=(2, 10),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    返回:
      enc_pos: (72, 2)  -> 3RB * 2导频符号 * 12子载波
      dec_pos: (144, 2) -> 当前RB * (n_sym-2)数据符号 * 12子载波
    坐标:
      每个位置是 (k_glob, n)，其中 k_glob 为窗口内的“全带局部子载波索引”，范围 [0, rb_sc*3-1]
      约定: 前一RB -> 区间 [0,11]，当前RB -> [12,23]，后一RB -> [24,35]
    """
    assert rb_sc == 12, "按 3RB 窗口默认每RB 12子载波；如需泛化请相应调整"
    pilot_set = set(pilot_syms)

    enc = []
    # encoder 72 个导频 token：顺序 = RB偏移(-1,0,+1) -> pilot_syms 顺序 -> k=0..11
    for rb_off in (-1, 0, +1):
        base = (rb_off + 1) * rb_sc  # -1->0, 0->12, +1->24
        for n in pilot_syms:
            for k in range(rb_sc):
                k_glob = base + k
                enc.append((k_glob, n))

    # decoder 仅当前RB的数据 token：顺序 = n(排除导频) -> k=0..11；k_glob 在 [12,23]
    dec = []
    base_cur = 1 * rb_sc  # 当前RB起点=12
    for n in range(n_sym):
        if n in pilot_set:
            continue
        for k in range(rb_sc):
            k_glob = base_cur + k
            dec.append((k_glob, n))

    enc_pos = torch.tensor(enc, dtype=torch.long)   # [72, 2]
    dec_pos = torch.tensor(dec, dtype=torch.long)   # [144, 2]

    # 形状断言
    assert enc_pos.shape == (3 * len(pilot_syms) * rb_sc, 2)  # 3*2*12=72
    assert dec_pos.shape == ((n_sym - len(pilot_syms)) * rb_sc, 2)  # (14-2)*12=144
    return enc_pos, dec_pos

def _split_enc_feats(enc_feats: torch.Tensor, num_rx: int):
    """
    enc_feats: [B3, 12, d_in]  with d_in = 4*num_rx + 2  (见上面的拼接顺序)
    returns:
      rx_r, rx_i: [B3, 12, num_rx]
      h_r,  h_i : [B3, 12, num_rx]
      xp_r, xp_i: [B3, 12, 1]         (pilot on each subcarrier)
    """
    B3, K, Din = enc_feats.shape
    #print(K)
    assert K == 12 and Din == 4 * num_rx + 2, "enc_feats 维度与 num_rx 不匹配"

    i0 = 0
    rx_r = enc_feats[:, :, i0:i0+num_rx]; i0 += num_rx
    rx_i = enc_feats[:, :, i0:i0+num_rx]; i0 += num_rx
    h_r  = enc_feats[:, :, i0:i0+num_rx]; i0 += num_rx
    h_i  = enc_feats[:, :, i0:i0+num_rx]; i0 += num_rx
    xp_r = enc_feats[:, :, i0:i0+1];      i0 += 1
    xp_i = enc_feats[:, :, i0:i0+1];      i0 += 1
    return rx_r, rx_i, h_r, h_i, xp_r, xp_i

def _split_dec_feats(enc_feats: torch.Tensor, num_rx: int):
    """
    enc_feats: [B3, 12, d_in]  with d_in = 4*num_rx + 2  (见上面的拼接顺序)
    returns:
      rx_r, rx_i: [B3, 12, num_rx]
      h_r,  h_i : [B3, 12, num_rx]
      xp_r, xp_i: [B3, 12, 1]         (pilot on each subcarrier)
    """
    B3, K, Din = enc_feats.shape
    #print(K)
    assert Din == 4 * num_rx, "dec_feats 维度与 num_rx 不匹配"

    i0 = 0
    rx_r = enc_feats[:, :, i0:i0+num_rx]; i0 += num_rx
    rx_i = enc_feats[:, :, i0:i0+num_rx]; i0 += num_rx
    h_r  = enc_feats[:, :, i0:i0+num_rx]; i0 += num_rx
    h_i  = enc_feats[:, :, i0:i0+num_rx]; i0 += num_rx
    return rx_r, rx_i, h_r, h_i

def _to_complex(r: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
    # 返回 complex64/complex32（跟随 r/i 的 dtype）
    return torch.complex(r, i)

# ----------------------------
# 2D sinusoidal positional encoding over (k, n)
# ----------------------------
class SinusoidalPE2D(nn.Module):
    def __init__(self, d_model: int, rb_sc: int = 12, n_sym: int = 14):
        super().__init__()
        self.d_model = d_model
        self.rb_sc = rb_sc
        self.n_sym = n_sym
        d_f = d_model // 2
        d_t = d_model - d_f
        self.d_f = d_f
        self.d_t = d_t
        self.register_buffer(
            "div_f",
            torch.exp(torch.arange(0, d_f, 2, dtype=torch.float32) * (-math.log(10000.0) / max(1, d_f)))
        )
        self.register_buffer(
            "div_t",
            torch.exp(torch.arange(0, d_t, 2, dtype=torch.float32) * (-math.log(10000.0) / max(1, d_t)))
        )

    def forward(self, pos_kn: torch.Tensor) -> torch.Tensor:
        L = pos_kn.size(0)
        k = pos_kn[:, 0].float() / max(1.0, (self.rb_sc - 1))
        n = pos_kn[:, 1].float() / max(1.0, (self.n_sym - 1))

        pe_f = torch.zeros(L, self.d_f, device=pos_kn.device)
        if self.d_f > 0:
            pe_f[:, 0::2] = torch.sin(k.unsqueeze(-1) * self.div_f)
            if self.d_f > 1:
                pe_f[:, 1::2] = torch.cos(k.unsqueeze(-1) * self.div_f)

        pe_t = torch.zeros(L, self.d_t, device=pos_kn.device)
        if self.d_t > 0:
            pe_t[:, 0::2] = torch.sin(n.unsqueeze(-1) * self.div_t)
            if self.d_t > 1:
                pe_t[:, 1::2] = torch.cos(n.unsqueeze(-1) * self.div_t)

        return torch.cat([pe_f, pe_t], dim=-1)  # [L, d_model]


# ----------------------------
# Conv-Stem: scatter -> Conv2d -> gather
# ----------------------------
class ConvStem2D(nn.Module):
    """
    Place token features onto a (n_sym, rb_sc) grid, run 2D conv, pick features back at token positions.
    This injects a small local 2D prior (denoising/smoothing) before Transformer.
    """
    def __init__(self, d_in: int, d_out: int, rb_sc: int = 12, n_sym: int = 14, hidden: int = None):
        super().__init__()
        self.rb_sc = rb_sc
        self.n_sym = n_sym
        hidden = hidden or max(d_in, d_out)
        self.net = nn.Sequential(
            nn.Conv2d(d_in, hidden, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, d_out, kernel_size=3, padding=1, bias=True),
        )

    def forward(self, feats: torch.Tensor, pos_kn: torch.Tensor) -> torch.Tensor:
        """
        feats:  [B3, L, C_in]
        pos_kn: [L, 2] (k, n)
        return: [B3, L, C_out]
        """
        B3, L, C = feats.shape
        device = feats.device
        grid = feats.new_zeros(B3, C, self.n_sym, self.rb_sc)  # [B3, C, n_sym, rb_sc]
        print(grid.shape)
        # scatter: fill only at token positions
        k = pos_kn[:, 0].long()
        n = pos_kn[:, 1].long()
        for b in range(B3):
            grid[b, :, n, k] = feats[b].transpose(0, 1)  # [C, n_sym, rb_sc] only at (n,k)
        print(grid[0,0,:,:])
        # 2D conv
        out = self.net(grid)  # [B3, C_out, n_sym, rb_sc]

        # gather back
        C_out = out.size(1)
        gathered = feats.new_empty(B3, L, C_out)
        for b in range(B3):
            gathered[b] = out[b, :, n, k].transpose(0, 1)  # [L, C_out]
        return gathered


# ----------------------------
# Relative position bias + neighborhood mask for cross-attn
# ----------------------------
class RelPosBias2D(nn.Module):
    def __init__(self, dec_pos: torch.Tensor, enc_pos: torch.Tensor,
                 max_df: int = 11, max_dt: int = 13):
        super().__init__()
        dk = dec_pos[:, 0].unsqueeze(1) - enc_pos[:, 0].unsqueeze(0)  # [144,24]
        dn = dec_pos[:, 1].unsqueeze(1) - enc_pos[:, 1].unsqueeze(0)  # [144,24]
        dk = dk.clamp(-max_df, max_df) + max_df
        dn = dn.clamp(-max_dt, max_dt) + max_dt
        index = dk * (2 * max_dt + 1) + dn
        self.register_buffer("index", index.long(), persistent=False)
        table_size = (2 * max_df + 1) * (2 * max_dt + 1)
        self.bias_table = nn.Parameter(torch.zeros(table_size))  # start with zeros

    def forward(self) -> torch.Tensor:
        return self.bias_table[self.index]  # [144,24]


def build_neighborhood_mask(dec_pos: torch.Tensor, enc_pos: torch.Tensor, K: int = 5) -> torch.Tensor:
    """
    Neighborhood mask for cross-attn.
    Allow attention only if |Δk| <= K (both pilot time slots allowed).
    Return a mask with 0.0 for allowed pairs and -inf for forbidden ones, shape [144, 24].
    """
    dk = (dec_pos[:, 0].unsqueeze(1) - enc_pos[:, 0].unsqueeze(0)).abs()  # [144,24]
    allowed = dk <= K
    mask = torch.zeros_like(dk, dtype=torch.float32)
    mask[~allowed] = float('-inf')
    return mask  # [144,24]


# ----------------------------
# Cross-Attn-only Decoder Block
# ----------------------------
class CrossAttnOnlyBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int,
                 dropout: float = 0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_q: torch.Tensor, mem_kv: torch.Tensor,
                attn_mask: torch.Tensor = None, key_padding_mask: torch.Tensor=None) -> torch.Tensor:
        # Cross-Attention
        res = x_q
        x_attn, _ = self.mha(x_q, mem_kv, mem_kv, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        x_q = self.norm1(res + self.dropout(x_attn))
        # FFN
        res = x_q
        x_ffn = self.ffn(x_q)
        x_q = self.norm2(res + self.dropout(x_ffn))
        return x_q


# ----------------------------
# Analytic (max-log) LLR head for square Gray QAM
# ----------------------------
def gray_labels_for_m(m: int) -> torch.Tensor:
    """Return Gray bit labels for values 0..m-1 as a tensor [m, bits]."""
    bits = int(math.log2(m))
    labels = torch.arange(m).unsqueeze(1).bitwise_xor(torch.arange(m).unsqueeze(1) >> 1)
    out = torch.stack([((labels.squeeze(1) >> b) & 1) for b in reversed(range(bits))], dim=1).float()
    return out  # [m, bits]


def build_square_qam_constellation(bits_per_symbol: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build normalized square QAM constellation with Gray coding along I and Q separately.
    Return points [M,2] (x,y) and bit labels [M, bits].
    """
    M = 1 << bits_per_symbol
    m_side = int(math.sqrt(M))
    assert m_side * m_side == M, "bits_per_symbol must form a square QAM"
    levels = torch.arange(-(m_side-1), m_side, 2, dtype=torch.float32)
    avg_pow_1d = (levels**2).mean()
    scale = (avg_pow_1d * 2).sqrt()  # E[x^2 + y^2] = 2*avg_pow_1d -> scale to unit avg power
    levels = levels / scale

    xs, ys = torch.meshgrid(levels, levels, indexing="xy")
    points = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=1)  # [M,2]

    gray_1d = gray_labels_for_m(m_side)  # [m_side, bits/2]
    gi = gray_1d.repeat_interleave(m_side, dim=0)
    gq = gray_1d.repeat(m_side, 1)
    labels = torch.cat([gi, gq], dim=1).float()  # [M, bits]
    return points, labels


class AnalyticLLRHead(nn.Module):
    """
    Map decoder hidden -> (z_re, z_im, log_sigma2), then compute max-log LLR vs square QAM constellation.
    """
    def __init__(self, d_model: int, bits_per_symbol: int):
        super().__init__()
        self.bits = bits_per_symbol
        self.M = 1 << bits_per_symbol
        self.param = nn.Linear(d_model, 3)  # -> [z_re, z_im, log_sigma2]
        pts, labels = build_square_qam_constellation(bits_per_symbol)
        self.register_buffer("const_points", pts)     # [M,2]
        self.register_buffer("const_bits", labels)    # [M,bits]

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        B3, L, _ = h.shape
        out = self.param(h)  # [B3, L, 3]
        z_re, z_im, log_s2 = out[..., 0], out[..., 1], out[..., 2].clamp(min=-8.0, max=8.0)
        z = torch.stack([z_re, z_im], dim=-1)  # [B3, L, 2]
        s2 = torch.exp(log_s2) + 1e-6

        diff = z.unsqueeze(2) - self.const_points.unsqueeze(0).unsqueeze(0)  # [B3,L,M,2]
        d2 = (diff ** 2).sum(dim=-1)  # [B3,L,M]

        llrs = []
        for b in range(self.bits):
            mask1 = self.const_bits[:, b] > 0.5  # [M]
            mask0 = ~mask1
            d2_1 = d2[..., mask1]  # [B3,L,M1]
            d2_0 = d2[..., mask0]  # [B3,L,M0]
            min0, _ = d2_0.min(dim=-1)
            min1, _ = d2_1.min(dim=-1)
            llr_b = (min0 - min1) / (s2 + 1e-6)  # [B3, L]
            llrs.append(llr_b.unsqueeze(-1))
        llrs = torch.cat(llrs, dim=-1)  # [B3, L, bits]
        return llrs

class HRefine1D(nn.Module):
    """
    对 h_est（频域 12 点）做轻量 refine：Conv1d+残差，沿频域建模。
    in/out: [B3, 12, 2*num_rx]  （最后会拆回成 [B3,12,num_rx] 的实部/虚部）
    """
    def __init__(self, num_rx: int, hidden: int = 64, depth: int = 2, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.c_in = 2 * num_rx
        self.c_h  = hidden
        self.depth = depth
        layers = []
        cin = self.c_in
        for d in range(depth):
            layers += [
                nn.Conv1d(cin, self.c_h, kernel_size, padding=kernel_size//2, groups=1, bias=True),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(self.c_h, self.c_in, kernel_size, padding=kernel_size//2, groups=1, bias=True),
            ]
            cin = self.c_in  # 每个残差块输出维度与输入相同

        self.blocks = nn.ModuleList(layers)

        # 残差缩放（ReZero 风格）+ 最后一层权重 0 初始化，稳定大模型训练
        self.gates = nn.ParameterList([nn.Parameter(torch.tensor(0.1, dtype=torch.float32)) 
                                       for _ in range(depth)])
        # zero-init 第二个conv（每个残差块的末层）权重，确保初始近似恒等
        for i in range(0, len(self.blocks), 4):
            nn.init.zeros_(self.blocks[i+3].weight)
            nn.init.zeros_(self.blocks[i+3].bias)

    def forward(self, h_ri: torch.Tensor) -> torch.Tensor:
        """
        h_ri: [B3, 12, 2*num_rx]
        returns: same shape
        """
        # TODO1： 对data部分进行split，datah_ri：【B3,12*12,2*num_rx】
        [B3, numRe, cin] = h_ri.shape
        if numRe == 12:
            num_chunk = 2
            num_resymbol = 6
        else:
            num_chunk = 12
            num_resymbol = 12
        x_split = torch.chunk(h_ri, num_chunk, dim=1)  # 拆分成两个 tensor，每个形状为 [B3, 6, 2*num_rx]
        # 第二步：将拆分后的两个 tensor 沿着第 1 个维度堆叠，得到 [B3, 2, 6, 2*num_rx] or data[[B3, 12, 12, 2*num_rx]]
        x_stacked = torch.stack(x_split, dim=1)
        # 第三步：将第 0 和第 1 个维度合并，得到 [2*B3, 6, 2*num_rx] or data[[12*B3, 12, 2*num_rx]]
        x_reshaped = x_stacked.reshape(-1, num_resymbol, self.c_in)
        x = x_reshaped.transpose(1, 2)  # -> [2*B3, C, 6] for Conv1d or data[[12*B3, C, 12]]
        y = x
        for b in range(self.depth):
            conv1 = self.blocks[4*b+0]
            act   = self.blocks[4*b+1]
            drop  = self.blocks[4*b+2]
            conv2 = self.blocks[4*b+3]
            z = conv2(drop(act(conv1(y))))
            y = y + self.gates[b] * z  # ReZero 残差
        y = y.transpose(1, 2)  # -> [2*B3, 6, 2*num_rx] or [12*B3, 12, 2*num_rx]
        y = y.reshape(B3, num_chunk, num_resymbol, self.c_in)
        out = y.reshape(B3, num_chunk*num_resymbol, self.c_in)
        
        return out

# ----------------------------
# Main Model (Scheme C+): Conv-Stem + Encoder + Sparse Cross-Attn Decoder + Dual LLR heads
# ----------------------------
@dataclass
class ModelCPlusConfig:
    d_in_pilot: int = 10 # 4*Rx+2
    d_in_data: int = 4 # 2*Rx
    d_model: int = 64
    d_model_data: int = 64
    nhead: int = 4
    nhead_data: int = 4
    num_enc_layers: int = 2
    num_dec_layers: int = 3
    dim_ff: int = 256
    dim_ff_data: int=256
    dropout: float = 0.1
    bits_per_symbol: int = 4
    neighbor_K: int = 2          # neighborhood radius in |Δk|
    stem_hidden: int = 64        # Conv-Stem hidden channels
    mix_init_logit: float = 0.0  # fusion weight init (sigmoid -> 0.5)


class ModelCPlus(nn.Module):
    def __init__(self, cfg: ModelCPlusConfig,
                 rb_sc: int = 12, n_sym: int = 14, pilot_syms=(2, 10), delta_zeros=1):
        super().__init__()
        self.cfg = cfg
        self.rb_sc = rb_sc
        self.n_sym = n_sym
        self.pilot_syms = pilot_syms
        self.numRx = 2
        # positions (fixed)
        enc_pos, dec_pos = build_re_positions(rb_sc, n_sym, pilot_syms)
        self.register_buffer("enc_pos", enc_pos, persistent=False)  # [24,2]
        self.register_buffer("dec_pos", dec_pos, persistent=False)  # [144,2]
        # zero pilot padding
        self.src_key_padding_mask = torch.zeros(rb_sc*len(pilot_syms), dtype=torch.bool)
        self.src_key_padding_mask[delta_zeros::2] = True
        #print(f'src_key_padding_mask', self.src_key_padding_mask)
        # Conv-Stem (scatter -> conv2d -> gather), keep channel dim = d_in
        #self.stem_enc = ConvStem2D(cfg.d_in_pilot, cfg.d_in_pilot, rb_sc, n_sym, hidden=cfg.stem_hidden)
        #self.stem_dec = ConvStem2D(cfg.d_in_data, cfg.d_in_data, rb_sc, n_sym, hidden=cfg.stem_hidden)
        self.h_refine = HRefine1D(num_rx=self.numRx, hidden=64, depth=2)
        
        # Input projections and 2D PE
        self.inp_proj_enc = nn.Linear(cfg.d_in_pilot, cfg.d_model)
        self.inp_proj_dec = nn.Linear(cfg.d_in_data, cfg.d_model)
        self.pe2d = SinusoidalPE2D(cfg.d_model, rb_sc, n_sym)
        self.dropout = nn.Dropout(cfg.dropout)
        
        # Encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.nhead,
            dim_feedforward=cfg.dim_ff, dropout=cfg.dropout,
            batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.num_enc_layers)
        self.encoder_lower_snr = nn.TransformerEncoder(enc_layer, num_layers=cfg.num_enc_layers)
        # Decoder (cross-attn only)
        self.dec_blocks = nn.ModuleList([
            CrossAttnOnlyBlock(cfg.d_model_data, cfg.nhead_data, cfg.dim_ff_data, cfg.dropout)
            for _ in range(cfg.num_dec_layers)
        ])

        # RelPos bias + neighborhood sparse mask
        self.relpos_bias = RelPosBias2D(self.dec_pos, self.enc_pos)
        self.register_buffer("neigh_mask", build_neighborhood_mask(self.dec_pos, self.enc_pos, cfg.neighbor_K),
                             persistent=False)  # [144,24]

        # Dual heads: learned and analytic
        self.learned_head = nn.Linear(cfg.d_model, cfg.bits_per_symbol)
        #self.analytic_head = AnalyticLLRHead(cfg.d_model, cfg.bits_per_symbol)

        # Temperature + fusion weight (sigmoid -> [0,1])
        #self.log_temp = nn.Parameter(torch.tensor(0.0))  # temp = exp(log_temp) >= 1e-6
        #self.mix_logit = nn.Parameter(torch.tensor(cfg.mix_init_logit))

    def add_pe(self, x: torch.Tensor, pos_kn: torch.Tensor) -> torch.Tensor:
        pe = self.pe2d(pos_kn).unsqueeze(0)  # [1,L,d]
        return x + pe

    def forward(self, enc_feats: torch.Tensor, dec_feats: torch.Tensor):
        """
        enc_feats: [B*3, 12, d_in] , din->{numRx*real_part, numRx*img_part, numRx*h_est_real_part, numRx*h_est_img_part, pilot_real_part, pilot_img_part}
        dec_feats: [B*3, 144, d_in]
        return: dict with 'logits', 'logits_learned', 'logits_analytic', 'temp', 'mix_weight'
        """
        device = enc_feats.device
        # TODO:加入导频纠正网络，使得mse(rx,f（h_est）*pilot)，即最小化估计导频和接收导频方差。使用f(h_est)作为导频特征送入后级网络。
        rx_r, rx_i, h_r, h_i, xp_r, xp_i = _split_enc_feats(enc_feats, num_rx=self.numRx)
        # 复数装配
        Y   = _to_complex(rx_r, rx_i)           # [B3, 12, num_rx]
        H0  = _to_complex(h_r,  h_i)            # [B3, 12, num_rx]
        Xp  = _to_complex(xp_r, xp_i).squeeze(-1)  # [B3, 12]  (1Tx)

        # 只把 h_est 送进 refine 子网（实/虚拼成通道）
        h_ri_in = torch.cat([h_r, h_i], dim=-1)             # [B3, 12, 2*num_rx]
        h_ri_out = self.h_refine(h_ri_in)                        # [B3, 12, 2*num_rx]
        h_ref_r, h_ref_i = torch.split(h_ri_out, self.numRx, dim=-1)
        H_ref = _to_complex(h_ref_r, h_ref_i)               # [B3, 12, num_rx]
        Y_hat = H_ref * Xp.unsqueeze(-1)  
        Y_hat_hat = H0 * Xp.unsqueeze(-1)  
        #print(f'Y_hat:',Y_hat.real[0,...])
        #print(f'Y:',Y.real[0,...])
        #print(f'Y_hat_hat',Y_hat_hat.real[0,...])
        loss = F.mse_loss(Y_hat.real, Y.real) + F.mse_loss(Y_hat.imag, Y.imag)
        loss_h_est = F.mse_loss(Y_hat_hat.real, Y.real) + F.mse_loss(Y_hat_hat.imag, Y.imag) #TODO2: 如果 loss 比 loss_h_est要小，增加一个低SNR的旁路分支，然后再进行encoder
        # replace h_est with h-refine
        if loss<loss_h_est: #loss refine < loss h_hat
            enc_feats[:,:,4:8] = h_ri_out
        # Project to d_model and add PE
        x_enc = self.inp_proj_enc(enc_feats)
        x_enc = self.add_pe(x_enc, self.enc_pos.to(device))
        x_enc = self.dropout(x_enc)
        #data h-refine
        drx_r, drx_i, dh_r, dh_i = _split_dec_feats(dec_feats, num_rx=self.numRx)
        dh_ri_in = torch.cat([dh_r, dh_i], dim=-1)
        dh_ri_out = self.h_refine(dh_ri_in)    
        #dh_ref_r, dh_ref_i = torch.split(dh_ri_out, self.numRx, dim=-1)
        #print(f"dh_ri_out:",dh_ri_out.shape)
        #print(f"dec_feats:",dec_feats.shape)
        if loss<loss_h_est: #loss refine < loss h_hat
            dec_feats[:,:,4:8] = dh_ri_out
        x_dec = self.inp_proj_dec(dec_feats)
        x_dec = self.add_pe(x_dec, self.dec_pos.to(device))
        x_dec = self.dropout(x_dec)

        # Encoder
        #src_key_padding_mask = self.src_key_padding_mask.unsqueeze(0).expand(x_enc.shape[0], -1)
        if loss<loss_h_est: #loss refine < loss h_hat
            mem = self.encoder_lower_snr(x_enc)  # [B3,24,d]
        else:
            mem = self.encoder(x_enc)  # [B3,24,d]
        # Build attn mask = neighborhood mask + relative position bias
        attn_bias = self.relpos_bias().to(device)  # [144,24]
        neigh = self.neigh_mask.to(device)         # [144,24]
        #print(neigh[:,0])
        attn_mask = attn_bias  # zeros on allowed, -inf on forbidden
        # Decoder blocks (cross-attn only)
        for blk in self.dec_blocks:
            x_dec = blk(x_dec, mem, attn_mask=attn_mask)

        # Heads
        logits_learned = self.learned_head(x_dec)      # [B3,144,bits]
        #logits_analytic = self.analytic_head(x_dec)    # [B3,144,bits]
        logits_analytic = None
        temp = None
        mix_w = None
        # Temperature and fusion
        #temp = torch.exp(self.log_temp).clamp(min=1e-3, max=100.0)
        #mix_w = torch.sigmoid(self.mix_logit)  # in [0,1], weight for analytic branch
        #logits = (1.0 - mix_w) * (logits_learned / temp) + mix_w * (logits_analytic / temp)
        #logits =  (logits_learned / temp)
        logits = logits_learned
        return {
            "logits": logits,
            "logits_learned": logits_learned,
            "logits_analytic": logits_analytic,
            "temp": temp,
            "mix_weight": mix_w,
            "mse_loss": loss,
            "loss_h_est": loss_h_est
        }



def llr_loss_focal(s_learn, bits,
                        margin_m=1.0, focal_gamma=2.0,
                        lam_focal=0.5, lam_brier=0.05,
                        topk_frac=0., tau=2):
    """
    s_learn: logits_learned, shape [B, 144, bits]
    bits:    {0,1}, shape   [B, 144, bits]
    """
    med = s_learn.abs().median().detach()

    # 自适应超参
    low = (med < tau)
    if low:
        print("median 小于 门限")
    margin_m = 0.2 if low else margin_m
    focal_gamma = 0.0 if low else focal_gamma
    lam_focal = 0.2 if low else lam_focal
    topk_frac = 0. if low else topk_frac
    lam_brier = 0.08 if low else lam_brier
    # to +/-1
    t = bits.mul(2.).sub(1.)         # {+1,-1}
    s = s_learn

    # 1) margin-logistic
    margin = F.softplus(-(t * s - margin_m))   # log(1+exp(-(t*s-m)))

    # 2) focal BCE
    bce = F.binary_cross_entropy_with_logits(s, bits, reduction='none')
    pt = torch.sigmoid(t * s)                   # confidence of the true class
    focal = (1. - pt).pow(focal_gamma) * bce
    
    # 3) brier (small weight)
    prob = torch.sigmoid(s)
    brier = (prob - bits).pow(2)

    lam_margin = (1-lam_focal) * pt.pow(focal_gamma)
    per_bit = lam_margin * margin + lam_focal * focal + lam_brier * brier  # [B,144,bits]

    # ---- Hard mining: keep only the lowest margins (most difficult) ----
    if topk_frac is not None and 0.0 < topk_frac < 1.0:
        with torch.no_grad():
            # difficulty score: smaller t*s -> harder
            diff = (t * s).reshape(-1)
            k = max(1, int(topk_frac * diff.numel()))
            idx = torch.topk(diff, k=k, largest=False).indices
        loss = per_bit.reshape(-1)[idx].mean()
    else:
        loss = per_bit.mean()
    return loss

def pilot_reconstruction_loss(enc_feats: torch.Tensor, num_rx: int, h_refine: HRefine1D):
    """
    enc_feats: [B3, 12, d_in]   （包含 rx, h_est, pilot；见 _split_enc_feats）
    returns:
      loss: 标量
      h_ref_r, h_ref_i: [B3, 12, num_rx]  （可作为后级 encoder 的导频特征）
    """
    rx_r, rx_i, h_r, h_i, xp_r, xp_i = _split_enc_feats(enc_feats, num_rx=num_rx)

    # 复数装配
    Y   = _to_complex(rx_r, rx_i)           # [B3, 12, num_rx]
    H0  = _to_complex(h_r,  h_i)            # [B3, 12, num_rx]
    Xp  = _to_complex(xp_r, xp_i).squeeze(-1)  # [B3, 12]  (1Tx)

    # 只把 h_est 送进 refine 子网（实/虚拼成通道）
    h_ri_in = torch.cat([h_r, h_i], dim=-1)             # [B3, 12, 2*num_rx]
    h_ri_out = h_refine(h_ri_in)                        # [B3, 12, 2*num_rx]
    h_ref_r, h_ref_i = torch.split(h_ri_out, num_rx, dim=-1)
    H_ref = _to_complex(h_ref_r, h_ref_i)               # [B3, 12, num_rx]

    # 预测导频接收: y_hat[b,k,rx] = H_ref[b,k,rx] * Xp[b,k]
    Y_hat = H_ref * Xp.unsqueeze(-1)                    # [B3, 12, num_rx]

    # 复数 MSE
    loss = F.mse_loss(Y_hat.real, Y.real) + F.mse_loss(Y_hat.imag, Y.imag)

    return loss, h_ref_r, h_ref_i


# ----------------------------
# Minimal training scaffold with dummy data
# ----------------------------
def demo_train_step(device="cpu"):
    torch.manual_seed(0)
    cfg = ModelCPlusConfig(
        d_in=8, d_model=64, nhead=4,
        num_enc_layers=2, num_dec_layers=3,
        dim_ff=128, dropout=0.1, bits_per_symbol=4,
        neighbor_K=2, stem_hidden=64, mix_init_logit=0.0
    )
    model = ModelCPlus(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # Dummy batch: B=2 -> B3=6
    B = 16
    B3 = B * 3
    d_in = cfg.d_in
    enc_feats = torch.randn(B3, 24, d_in, device=device)
    dec_feats = torch.randn(B3, 144, d_in, device=device)

    # Forward
    out = model(enc_feats, dec_feats)  # dict
    logits = out["logits"]

    # Dummy labels (random bits)
    target_bits = torch.randint(0, 2, (B3, 144, cfg.bits_per_symbol), device=device).float()

    # BCE loss
    loss_main = F.binary_cross_entropy_with_logits(logits, target_bits)

    # Optional auxiliary consistency loss between branches (small weight)
    loss_cons = F.mse_loss(out["logits_learned"], out["logits_analytic"].detach())

    loss = loss_main + 0.05 * loss_cons
    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

    info = {
        "loss": float(loss.item()),
        "loss_main": float(loss_main.item()),
        "loss_cons": float(loss_cons.item()),
        "temp": float(out["temp"].item()),
        "mix_weight": float(out["mix_weight"].item()),
        "logits_shape": tuple(logits.shape),
    }
    return info


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    info = demo_train_step(device=device)
    print("Demo train step info:", info)
