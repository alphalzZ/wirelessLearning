# transformer_ofdm_schemeC_demo.py
# -*- coding: utf-8 -*-
import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path

project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)
# ----------------------------
# Helpers: token位置索引（RE级）
# ----------------------------
def build_re_positions(rb_sc: int = 12, n_sym: int = 14,
                       pilot_syms=(2, 10)) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    生成 Encoder/Decoder 的 RE 级 token 的 (k, n) 位置列表。
    - k: 0..rb_sc-1  （频域子载波索引）
    - n: 0..n_sym-1  （时域OFDM符号索引）
    Encoder: 只包含导频时隙中的 RE（24 个）
    Decoder: 只包含数据时隙中的 RE（144 个）
    返回: enc_pos [24,2], dec_pos [144,2]
    """
    pilot_set = set(pilot_syms)
    enc = []
    dec = []
    # 顺序使用 "时间优先，再频域"（方便人读；不影响功能）
    for n in range(n_sym):
        for k in range(rb_sc):
            if n in pilot_set:
                enc.append((k, n))
            else:
                dec.append((k, n))
    enc_pos = torch.tensor(enc, dtype=torch.long)  # [24, 2]
    dec_pos = torch.tensor(dec, dtype=torch.long)  # [144, 2]
    # assert enc_pos.shape[0] == 24 and dec_pos.shape[0] == 144
    return enc_pos, dec_pos


# ----------------------------
# 2D 正余弦位置编码（频/时）
# ----------------------------
class SinusoidalPE2D(nn.Module):
    """
    对 (k, n) 做 2D 正余弦编码，并投到 d_model 维度（freq一半 + time一半）。
    可与输入线性投影相加（Add）使用。
    """
    def __init__(self, d_model: int, rb_sc: int = 12, n_sym: int = 14):
        super().__init__()
        self.d_model = d_model
        self.rb_sc = rb_sc
        self.n_sym = n_sym

        d_f = d_model // 2
        d_t = d_model - d_f
        self.d_f = d_f
        self.d_t = d_t

        # div_term 与标准 Transformer 一致（log 10000），独立给频/时两轴
        self.register_buffer(
            "div_f",
            torch.exp(torch.arange(0, d_f, 2, dtype=torch.float32) * (-math.log(10000.0) / max(1, d_f)))
        )
        self.register_buffer(
            "div_t",
            torch.exp(torch.arange(0, d_t, 2, dtype=torch.float32) * (-math.log(10000.0) / max(1, d_t)))
        )

    def forward(self, pos_kn: torch.Tensor) -> torch.Tensor:
        """
        pos_kn: [L, 2], 其中 [:,0]=k ∈ [0, rb_sc-1], [:,1]=n ∈ [0, n_sym-1]
        返回: pe [L, d_model]
        """
        L = pos_kn.size(0)
        k = pos_kn[:, 0].float() / max(1.0, (self.rb_sc - 1))    # 归一化到 [0,1]
        n = pos_kn[:, 1].float() / max(1.0, (self.n_sym - 1))    # 归一化到 [0,1]

        pe_f = torch.zeros(L, self.d_f, device=pos_kn.device)
        if self.d_f > 0:
            # 偶数维 sin, 奇数维 cos
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
# 2D 相对位置偏置（对 Cross-Attn 的 attn_mask 贡献）
# ----------------------------
class RelPosBias2D(nn.Module):
    """
    针对固定的 Decoder(144) ↔ Encoder(24) 的 (k,n) 对生成可学习的 2D 相对位置偏置。
    将 Δk, Δn 映射为一个标量偏置，填入 attn_mask: [Lq, Lk]。
    —— 简化版本：所有头共享同一偏置矩阵（够用）。
    """
    def __init__(self, dec_pos: torch.Tensor, enc_pos: torch.Tensor,
                 max_df: int = 11, max_dt: int = 13):
        super().__init__()
        # 计算 Δk、Δn
        # dec_pos: [144,2], enc_pos: [24,2]
        dk = dec_pos[:, 0].unsqueeze(1) - enc_pos[:, 0].unsqueeze(0)  # [144,24]
        dn = dec_pos[:, 1].unsqueeze(1) - enc_pos[:, 1].unsqueeze(0)  # [144,24]
        # clamp 到预设范围
        dk = dk.clamp(-max_df, max_df) + max_df  # -> [0..2*max_df]
        dn = dn.clamp(-max_dt, max_dt) + max_dt  # -> [0..2*max_dt]
        index = dk * (2 * max_dt + 1) + dn  # 合并为单索引
        index = index.long()  # [144,24]
        self.register_buffer("index", index, persistent=False)

        table_size = (2 * max_df + 1) * (2 * max_dt + 1)
        self.bias_table = nn.Parameter(torch.zeros(table_size))  # 初始化0较稳

    def forward(self) -> torch.Tensor:
        # 取表 -> [144,24]
        attn_bias = self.bias_table[self.index]
        return attn_bias  # 作为 MultiheadAttention 的 attn_mask（加到logits上）


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
                attn_mask: torch.Tensor = None) -> torch.Tensor:
        """
        x_q:  [B, Lq, d]
        mem:  [B, Lk, d]
        attn_mask: [Lq, Lk]（会加到所有head的注意力logits上）
        """
        # Cross-Attention
        x_res = x_q
        x_q, _ = self.mha(x_q, mem_kv, mem_kv, attn_mask=attn_mask)
        x_q = self.norm1(x_res + self.dropout(x_q))
        # FFN
        x_res = x_q
        x_q = self.ffn(x_q)
        x_q = self.norm2(x_res + self.dropout(x_q))
        return x_q


# ----------------------------
# 主模型（方案C）
# ----------------------------
@dataclass
class ModelCConfig:
    d_in: int = 4           # 输入特征维度（例如 Re, Im, |·|, angle）
    d_model: int = 64
    nhead: int = 4
    num_enc_layers: int = 2
    num_dec_layers: int = 3
    dim_ff: int = 128
    dropout: float = 0.1
    bits_per_symbol: int = 4  # 例如16-QAM -> 4bit/符号


class ModelC(nn.Module):
    """
    Encoder: 24 pilot tokens / RB
    Decoder: 144 data tokens / RB
    输出: [B*3, 144, bits] —— 每个数据RE的bit级logit（≈ LLR）
    """
    def __init__(self, cfg: ModelCConfig,
                 rb_sc: int = 12, n_sym: int = 14, pilot_syms=(2, 11)):
        super().__init__()
        self.cfg = cfg
        self.rb_sc = rb_sc
        self.n_sym = n_sym
        self.pilot_syms = pilot_syms

        # 固定 token 位置（RE级）
        enc_pos, dec_pos = build_re_positions(rb_sc, n_sym, pilot_syms)
        self.register_buffer("enc_pos", enc_pos, persistent=False)  # [24,2]
        self.register_buffer("dec_pos", dec_pos, persistent=False)  # [144,2]

        # 位置编码器（2D）
        self.pe2d = SinusoidalPE2D(cfg.d_model, rb_sc, n_sym)

        # 输入线性投影到 d_model
        self.inp_proj_enc = nn.Linear(cfg.d_in, cfg.d_model)
        self.inp_proj_dec = nn.Linear(cfg.d_in, cfg.d_model)

        # Encoder: 标准 TransformerEncoder（自注意力在 24 个导频token上）
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.nhead,
            dim_feedforward=cfg.dim_ff, dropout=cfg.dropout,
            batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.num_enc_layers)

        # Decoder: 仅 Cross-Attn block 堆叠
        self.dec_blocks = nn.ModuleList([
            CrossAttnOnlyBlock(cfg.d_model, cfg.nhead, cfg.dim_ff, cfg.dropout)
            for _ in range(cfg.num_dec_layers)
        ])

        # 2D 相对位置偏置（用于 cross-attn）
        self.relpos_bias = RelPosBias2D(self.dec_pos, self.enc_pos)

        # 输出头：逐 RE 的 bit 级 logit（≈ LLR）
        self.llr_head = nn.Linear(cfg.d_model, cfg.bits_per_symbol)

        # Dropout
        self.dropout = nn.Dropout(cfg.dropout)

    def add_positional_encoding(self, x: torch.Tensor, pos_kn: torch.Tensor) -> torch.Tensor:
        """
        x: [B3, L, d_model]
        pos_kn: [L,2]
        """
        pe = self.pe2d(pos_kn).unsqueeze(0)  # [1, L, d_model]
        return x + pe  # 直接相加

    def forward(self, enc_feats: torch.Tensor, dec_feats: torch.Tensor) -> torch.Tensor:
        """
        enc_feats: [B*3, 24, d_in]   —— 导频 RE 的特征（dummy）
        dec_feats: [B*3, 144, d_in]  —— 数据 RE 的特征（dummy）
        返回: llr_logits [B*3, 144, bits_per_symbol]
        """
        B3 = enc_feats.size(0)
        device = enc_feats.device

        # 线性投影 + 2D 位置编码
        x_enc = self.inp_proj_enc(enc_feats)                    # [B3, 24, d_model]
        #x_enc = self.add_positional_encoding(x_enc, self.enc_pos.to(device))
        x_enc = self.dropout(x_enc)

        x_dec = self.inp_proj_dec(dec_feats)                    # [B3, 144, d_model]
        #x_dec = self.add_positional_encoding(x_dec, self.dec_pos.to(device))
        x_dec = self.dropout(x_dec)

        # Encoder
        mem = self.encoder(x_enc)                               # [B3, 24, d_model]

        # Cross-Attn-only Decoder（可学习 2D 相对位置偏置 -> attn_mask）
        attn_mask = self.relpos_bias().to(device)               # [144, 24]
        for blk in self.dec_blocks:
            x_dec = blk(x_dec, mem, attn_mask=attn_mask)

        # LLR头（bit级logit）
        llr_logits = self.llr_head(x_dec)                       # [B3, 144, bits]
        return llr_logits

def save_full(model: nn.Module, path: str):
    torch.save(model.state_dict(), path)

def load_full(model: nn.Module, path: str, device):
    sd = torch.load(path, map_location=device)
    model.load_state_dict(sd, strict=True)


# ====== 实现：ClipStdPerBit ======
class ClipStdPerBit:
    """
    对标签 L（最后一维为 bit-plane）做：
      1) 对称截断到 [-Lmax_k, Lmax_k]
      2) 逐 bit-plane 标准化 (减均值 / 除标准差)
    用法：
      clipper = ClipStdPerBit(Lmax_per_bit=[8,8,6,6])
      clipper.fit(L_train)       # 仅用训练集
      Z = clipper.forward(L)     # 训练时把标签转到标准化域
      L_hat = clipper.inverse(Z) # 推理/评估时还原到 LLR 域（截断域）
    """
    def __init__(self, Lmax_per_bit, eps=1e-6):
        self.bits_per_symb = len(Lmax_per_bit)
        self.Lmax_vec = torch.as_tensor(Lmax_per_bit, dtype=torch.float32)
        self.eps = eps
        self.mu = None      # shape: [1,...,1,Nbps]
        self.sigma = None   # shape: [1,...,1,Nbps]
        self.mu_vec = None      # shape: [Nbps]
        self.sigma_vec = None   # shape: [Nbps]

    def _broadcast_Lmax(self, x):
        return self.Lmax_vec.to(x.device, x.dtype).view(*([1]*(x.ndim-1)), -1)

    def fit(self, L_train): 
        # L_train shape [Batch, Nre*bitsPerSymb]
        self.B, llr_len = L_train.shape
        L_train = L_train.view(self.B, -1, self.bits_per_symb)
        # 只用训练集拟合标准化参数
        Lmax = self._broadcast_Lmax(L_train)
        Lc = torch.clamp(L_train, -Lmax, Lmax)
        reduce_dims = tuple(range(Lc.ndim - 1))  # 除最后一维（bit-plane）外都做统计
        self.mu = Lc.mean(dim=reduce_dims, keepdim=True)
        self.sigma = Lc.std(dim=reduce_dims, keepdim=True).clamp_min(self.eps)
        # 保存 1D 版本，便于查看/加权
        self.mu_vec = self.mu.squeeze().detach()
        self.sigma_vec = self.sigma.squeeze().detach()

    def forward(self, L):
        assert self.mu is not None, "Call fit() before forward()."
        L = L.view(self.B, -1, self.bits_per_symb)
        Lmax = self._broadcast_Lmax(L)
        Lc = torch.clamp(L, -Lmax, Lmax)
        Lc = (Lc - self.mu) / self.sigma
        return Lc.view(self.B,-1)

    def inverse(self, Z):
        assert self.mu is not None, "Call fit() before inverse()."
        Z = Z.view(self.B, -1, self.bits_per_symb)
        Z =  Z * self.sigma + self.mu  # 还原到“已截断”的 LLR 域
        return Z.view(self.B,-1)
    
if __name__ == "__main__":
    clipper = ClipStdPerBit(Lmax_per_bit=[8,8,6,6])
    dummyLLR= torch.randn(4,256)
    clipper.fit(dummyLLR)
    Z_train = clipper.forward(dummyLLR) 
    print(Z_train[0,0:10])
    L_pred_clipped_domain = clipper.inverse(dummyLLR)
    print(L_pred_clipped_domain[0,0:10])