"""
在线数据生成器模块
作者：AI助手
日期：2025-06-27
"""
import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader
import sys
from typing import Iterator,Tuple, Generator
from pathlib import Path

# 添加项目根目录到Python路径
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.TransformerRx.basic_modules import ModelCConfig, ModelC, load_full
from src.TransformerRx.transformer_ofdm_schemeC_plus_demo import *
from src.config import OFDMConfig
from src.ofdm_tx import ofdm_tx, compute_k, add_timing_offset_and_freq_offset
from src.channel import awgn_channel, multipath_channel
from src.ofdm_rx import (
    remove_cp_and_fft,
    estimate_timing_offset,
    estimate_frequency_offset,
    compensate_frequency_offset,
    compensate_timing_offset,
    estimate_channel,
    noise_var_estimate,
    noise_covariance_estimate,
    channel_equalization,
    qam_demodulation
    )
    
def pilot_triplet_view(x, pad_mode='edge'):
    """
    将每个“目标RB”的导频记忆扩展为 [前一RB, 当前RB, 后一RB] 三个RB的导频合并，
    用于 encoder 的 cross-attn 记忆。

    Args:
        x: Tensor, 形状 (B, F, S, K)
            B: batch
            F: feature通道（如 Re/Im(H_LS), σ̂² 等）
            S: OFDM符号数（例如14）
            K: 全带子载波数 = n_RB * 12
        pad_mode: 边界处理方式:
            'edge'  -> 用边界RB重复填充 prev/next（最实用）


    Returns:
        enc_feats: (B * n_RB, 72, F)
            对每个目标RB，依次拼接 [prevRB 24个导频token, currRB 24, nextRB 24] 共72个token
            token顺序：对每个RB内，先按 pilot_syms 顺序，再按子载波k=0..11
    """
    n_sc_per_rb = 12
    B, F, S, K = x.size()
    assert K % n_sc_per_rb == 0, "K 必须是 12 的整数倍"
    n_RB = K // n_sc_per_rb

    # 变形到 (B, F, S, n_RB, 12)
    x_pilot = x.view(B, F, S, n_RB, n_sc_per_rb)  # (B,F,S,R,Kp)

    # 调维到 (B,R,2,Kp,F)
    xp = x_pilot.permute(0, 3, 2, 4, 1).contiguous()

    # 构造 3RB 窗口的索引（prev, curr, next）
    idx = torch.arange(n_RB, device=x.device)
    idx_prev = idx - 1
    idx_curr = idx
    idx_next = idx + 1

    if pad_mode == 'edge':
        idx_prev = idx_prev.clamp(0, n_RB - 1)
        idx_next = idx_next.clamp(0, n_RB - 1)

    # 取每个目标RB对应的 prev/curr/next -> (B,R,2,Kp,F) 各一份
    x_prev = xp[:, idx_prev]     # (B,R,2,Kp,F)
    x_curr = xp[:, idx_curr]
    x_next = xp[:, idx_next]

    # 堆叠邻居维度 -> (B,R,3,2,Kp,F)
    x_win = torch.stack([x_prev, x_curr, x_next], dim=2)

    # 展开为 (B*R, 3*2*Kp, F) = (B*n_RB, 72, F)
    enc_feats = x_win.reshape(B * n_RB, 3 * 2 * n_sc_per_rb, F).contiguous()
    return enc_feats    


def rb_view(x, pilotFlag=0):
    """
    x: (B batch, F n_feature, S n_symbol, K n_subcarrier)  其中 K = n_RB * K_per
    return: (B*n_RB,  S* K_per, F)
    """
    n_sc_per_rb = 12 if not pilotFlag else 6
    B, F, S, K = x.size()
    assert K % n_sc_per_rb == 0
    n_RB = K // n_sc_per_rb
    
    # 1. 把最后一维切成 n_RB 块
    x = x.view(B, F, S, n_RB, n_sc_per_rb)          # (B, F, S, n_RB, K_per)
    # 2. 把 n_RB 移到第 0 维并合并到 batch
    x = x.permute(0, 3, 2, 4, 1).contiguous() # (B, n_RB, F, S, K_per)
    x = x.view(B * n_RB,  S * n_sc_per_rb, F)         # (B*n_RB, S * K_per, F)
    return x

def rb_view_reverse(x, batch_size):
    """
    x:(B*n_RB,  S* K_per, bits_per_symb)
    return: (B, S, K, bits_per_symb)
    """
    B = batch_size
    n_sc_per_rb = 12
    n_data_sym = 12
    B3, SK_pre, BITS_PER_SYMB = x.size()
    n_RB = B3//B

    x = x.view(B, n_RB, n_data_sym, n_sc_per_rb, BITS_PER_SYMB)
    x = x.permute(0,2,1,3,4).contiguous() #(B,symb,K_per,n_RB,bits_per_symb)
    x = x.view(B,n_data_sym* n_RB* n_sc_per_rb* BITS_PER_SYMB)
    return x

def create_data(cfg: OFDMConfig, batch_size: int) -> Generator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
                                                                None, None]:
    """
    在线生成一批训练数据。

    Args:
        cfg (OFDMConfig): OFDM系统配置。
        batch_size (int): 每批数据的样本数。

    Yields:
        Generator[Tuple[np.ndarray, np.ndarray], None, None]: 
        一个生成器，每次产出一批训练数据。
        元组的第一个元素是接收到的时域信号 (X),形状为 (batch_size, num_tx_ant, signal_len)。
        元组的第二个元素是原始的发送bits (y),形状为 (batch_size, num_bits)。
        元组的第三个元素是llr,形状为(batch_size, num_bits)。
    """
    while True:
        
        batch_pilot = []
        batch_data = []
        batch_bits = []
        batch_llr = []
        for _ in range(batch_size):
            # 1. 生成随机比特
            k = compute_k(cfg, cfg.code_rate)
            tx_bits = np.random.randint(0, 2, k)

            # 2. OFDM发送端处理，生成时域信号和频域符号
            tx_signal, freq_symbols = ofdm_tx(tx_bits, cfg)

            # 3. 通过信道
            if cfg.channel_type == 'awgn':
                rx_signal = awgn_channel(tx_signal, num_rx=cfg.num_rx_ant, num_tx=cfg.num_tx_ant)
            elif cfg.channel_type == 'multipath':
                rx_signal, h = multipath_channel(tx_signal, num_rx=cfg.num_rx_ant, num_tx=cfg.num_tx_ant)
            else:
                # 可根据需要扩展其他信道模型
                raise ValueError(f"不支持的信道类型: {cfg.channel_type}")
            
            # 4. (可选) 添加频偏和时偏
            signal = add_timing_offset_and_freq_offset(rx_signal, cfg)

            if signal.ndim == 1:

                signal = signal[None, :]

            num_ant = signal.shape[0]

            # 1. 移除循环前缀并进行FFT
            rx_symbols = remove_cp_and_fft(signal, cfg)
            
            # 2. 使用导频进行频偏估计和补偿
            offset = cfg.get_subcarrier_offset()
            pilot_symbol_indices = cfg.get_pilot_symbol_indices()
            pilot_symbols = cfg.get_pilot_symbols(pilot_symbol_indices)#排列顺序为第一根天线第一个DMRS，第二根天线，第一个DMRS，第一根天线，第二个DMRS，...
            pilot_symbols = pilot_symbols.reshape(cfg.num_tx_ant,len(pilot_symbol_indices), -1)  # (num_tx_ant, num_pilots, n_pilots)
            pilot_indices = cfg.get_pilot_indices() - offset
            pilot_symbols_include_zeros = np.zeros((cfg.num_tx_ant, len(pilot_symbol_indices), cfg.n_subcarrier),dtype=rx_symbols.dtype)
            #pilot_symbols_include_zeros[...,pilot_indices] = pilot_symbols
            for tx in range(cfg.num_tx_ant):
                for sym in range(len(pilot_symbol_indices)):
                    pilot_symbols_include_zeros[tx, sym, pilot_indices] = pilot_symbols[tx, sym, :]
            # print(f'pilot_symbols_include_zeros shape{pilot_symbols_include_zeros.shape}')
            est_timing = []
            est_freq_offset = []
            for a in range(num_ant):
                est_timing.append(
                    estimate_timing_offset(rx_symbols[a], pilot_symbols, pilot_indices, cfg)
                )
                est_freq_offset.append(
                    estimate_frequency_offset(rx_symbols[a], pilot_symbols, pilot_indices, cfg)
                )
            if cfg.num_tx_ant == 1:
                est_time = np.array(est_timing)[:,None]
                est_freq = np.array(est_freq_offset)[:,None]
            else:
                est_time = np.array(est_timing)
                est_freq = np.array(est_freq_offset)
            est_timing = np.mean(est_time,axis=1)
            est_freq_offset = np.mean(est_freq, axis=1)

            # 逐层逐天线补偿频偏
            comp_ant = [
                compensate_frequency_offset(
                    signal[a],
                    est_freq_offset[a],
                    cfg,
                )
                for a in range(num_ant)
            ]
            signal_freq_comp = np.stack(comp_ant, axis=0)

            # CP 移除与FFT
            rx_symbols = remove_cp_and_fft(signal_freq_comp, cfg)

            # 逐层逐天线补偿时延
            comp_ant = [
                compensate_timing_offset(
                    rx_symbols[a],
                    est_timing[a],
                    cfg,
                )
                for a in range(num_ant)
            ]

            signal_timing = np.stack(comp_ant, axis=0) # (num_ant, num_symbols, n_subcarrier)
            num_layer = cfg.num_tx_ant
            signal_timing_saved = signal_timing
            h_est_layer = []
            for l in range(num_layer):#多layer下倾向联合信道估计
                h_est_ant = []
                for a in range(num_ant):
                    h_est_tmp = estimate_channel(signal_timing[a], cfg, pilot_symbols[l], pilot_indices)
                    h_est_ant.append(h_est_tmp)
                h_est = np.stack(h_est_ant, axis=0)
                h_est_layer.append(h_est)
            h_est = np.stack(h_est_layer, axis=0) # (num_layer, num_ant, num_symbols, n_subcarrier)
            noise_ant = []
            power_ant = []
            for ant in range(num_ant):
                n_l, p_l = noise_var_estimate(signal_timing[ant], h_est[:,ant,...], cfg, pilot_symbols, pilot_indices)
                if cfg.display_est_result:
                    sinr = 10 * np.log10(np.mean(p_l) / np.mean(n_l))
                    print(f"layer {l+1} 天线 {ant+1} 估计的SINR: {sinr :.2f} dB")
                noise_ant.append(n_l)
                power_ant.append(p_l)
            noise_var = np.stack(noise_ant, axis=0)
            RxPower = np.stack(power_ant, axis=0)
            noise_cov = [[] for _ in range(num_layer)]
            if cfg.equ_method == 'irc':
                noise_cov_list = []
                for l in range(num_layer):
                    noise_cov_layer = noise_covariance_estimate(signal_timing, h_est[l], cfg, pilot_symbols[l], pilot_indices)
                    noise_cov_list.append(noise_cov_layer)
                noise_cov = np.stack(noise_cov_list, axis=0)
            
            rx_combined_list = []
            if cfg.equ_method == 'irc' or cfg.equ_method == 'mrc':
                for l in range(num_layer):
                    eq = channel_equalization(signal_timing, h_est[l], noise_var[l], cfg, noise_cov[l])
                    if eq.ndim == 2:
                        eq = eq[None, :, :]
                    rx_combined_list.append(np.mean(eq, axis=0))
                rx_combined = np.stack(rx_combined_list, axis=0)
            # 4. QAM 解调
            data_symbol_indices = cfg.get_data_symbol_indices()

            bits_list = []
            llr_list = []
            for l in range(num_layer):
                llr = qam_demodulation(
                    rx_combined[l, data_symbol_indices],
                    cfg.mod_order,
                    return_llr=True,
                    noise_var=float(np.mean(noise_var[l] / RxPower[l])),
                )
                llr_list.append(llr)
                tmp_bits = (llr < 0).astype(np.int8)
                bits_list.append(tmp_bits)
            llr_rx = np.stack(llr_list, axis=0)
            bits_rx = np.stack(bits_list, axis=0)
            llr_mask = bits_rx==tx_bits
            # complex signal power normalize
            h_Power_scale = np.mean(np.abs(h_est)**2) 
            h_est = h_est / np.sqrt(h_Power_scale)
            #print(f'h_Power_scale:',h_Power_scale)
            signal_Power_scale = np.mean(np.abs(signal_timing_saved)**2) 
            #print(f'signal_Power_scale:',signal_Power_scale)
            signal_timing_saved = signal_timing_saved / np.sqrt(signal_Power_scale)
            rx_real_part = np.real(signal_timing_saved)
            rx_imag_part = np.imag(signal_timing_saved)
            h_real_part = np.real(h_est[0,...])
            h_imag_part = np.imag(h_est[0,...])
            pilot_Power_scale = np.mean(np.abs(pilot_symbols_include_zeros[...,::2])**2)
            #print(f'pilot_Power_scale:', pilot_Power_scale)
            pilot_symbols_include_zeros = pilot_symbols_include_zeros / np.sqrt(pilot_Power_scale)
            pilot_real_prat = np.real(pilot_symbols_include_zeros)
            pilot_imag_part = np.imag(pilot_symbols_include_zeros)
            pilots_indx = cfg.get_pilot_symbol_indices()
            data_indx = cfg.get_data_symbol_indices()
            #导频数据的非导频位置信息置0
            dim1,dim2,dim3 = signal_timing_saved.shape
            
            
            mask = np.zeros(dim3, dtype=bool)
            mask[pilot_indices] = True
            #print(f'pilots_indx:',tmp.shape)
            mask_3d = mask[np.newaxis, np.newaxis, :]  # 形状: (1, 1, dim3)
            batch_pilot.append(np.concatenate((rx_real_part[...,pilots_indx,::2], rx_imag_part[...,pilots_indx,::2],
                                                h_real_part[...,pilots_indx,::2], h_imag_part[...,pilots_indx,::2],
                                                pilot_real_prat[...,::2], pilot_imag_part[...,::2]), axis=0, dtype=np.float32)) #TO:加入导频纠正，使得rx*w*h_est=pilot
            batch_data.append(np.concatenate((rx_real_part[...,data_indx,:], rx_imag_part[...,data_indx,:],
                                              h_real_part[...,data_indx,:], h_imag_part[...,data_indx,:]), axis=0, dtype=np.float32))  # 将实部和虚部堆叠成最后一维 #TODO：去掉信道估计结果，仅使用接收信号
            batch_bits.append(tx_bits)  # 将比特转换为一行
            batch_llr.append(np.squeeze(llr_rx*llr_mask))

        # 将列表转换为numpy数组
        pilot = np.array(batch_pilot,dtype=np.float32)
        data = np.array(batch_data,dtype=np.float32)  # (batch_size, 2*num_rx_ant, num_symbols, n_subcarrier)
        # print(pilot.shape)
        # print(data.shape)
        bits = np.array(batch_bits,dtype=np.float32)  # (batch_size, k)
        llr = np.array(batch_llr,dtype=np.float32)
        yield pilot, data, bits, llr

class OFDMTorchStreamDataset(IterableDataset):
    """
    无限流式数据集，每次迭代返回一个 batch。
    输出：
        x : [B, 4*num_rx_ant, num_symbols, n_subcarrier]  float32
        y : [B, k]                                           float32
    """

    def __init__(self, cfg: OFDMConfig, batch_size: int):
        super().__init__()
        self.cfg = cfg
        self.batch_size = batch_size
        self.k = compute_k(cfg, cfg.code_rate)

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        while True:                       # 无限循环 -> 无限数据流
            # create_data 返回 (x_np, y_np)
            generator = create_data(self.cfg, self.batch_size)
            x_pilot, x_data, y_bits, z_llr = next(generator)
            # 转成 torch.Tensor

            x_pilot_torch = rb_view(torch.from_numpy(x_pilot.astype(np.float32)),1)
            x_data_torch = rb_view(torch.from_numpy(x_data.astype(np.float32)))
            y_bits_torch = torch.from_numpy(y_bits.astype(np.float32))
            z_llr_torch = torch.from_numpy(z_llr.astype(np.float32))
            yield x_pilot_torch, x_data_torch, y_bits_torch, z_llr_torch

def create_pytorch_dataloader(cfg: OFDMConfig,
                              batch_size: int,
                              num_workers: int = 0,
                              pin_memory: bool = True) -> DataLoader:
    """
    生成 PyTorch 的 DataLoader，等价于原 tf.data.Dataset。
    用法：
        for x, y in loader:
            ...
    """
    dataset = OFDMTorchStreamDataset(cfg, batch_size)
    loader = DataLoader(
        dataset,
        batch_size=None,      # 已经在 dataset 里组好 batch
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0)
    )
    return loader

def ofdm_transformer_rx(model_weights_path, signal:np.ndarray, cfg:OFDMConfig, run_onnx_flag=False, plus_falg=True):
    
    if signal.ndim == 1:
        signal = signal[None, :]

    num_ant = signal.shape[0]

    # 1. 移除循环前缀并进行FFT
    rx_symbols = remove_cp_and_fft(signal, cfg)
    
    # 2. 使用导频进行频偏估计和补偿
    offset = cfg.get_subcarrier_offset()
    pilot_symbol_indices = cfg.get_pilot_symbol_indices()
    pilot_symbols = cfg.get_pilot_symbols(pilot_symbol_indices)#排列顺序为第一根天线第一个DMRS，第二根天线，第一个DMRS，第一根天线，第二个DMRS，...
    pilot_symbols = pilot_symbols.reshape(cfg.num_tx_ant,len(pilot_symbol_indices), -1)  # (num_tx_ant, num_pilots, n_pilots)
    pilot_indices = cfg.get_pilot_indices() - offset
    pilot_symbols_include_zeros = np.zeros((cfg.num_tx_ant, len(pilot_symbol_indices), cfg.n_subcarrier),dtype=rx_symbols.dtype)
    #pilot_symbols_include_zeros[...,pilot_indices] = pilot_symbols
    for tx in range(cfg.num_tx_ant):
        for sym in range(len(pilot_symbol_indices)):
            pilot_symbols_include_zeros[tx, sym, pilot_indices] = pilot_symbols[tx, sym, :]
    est_timing = []
    est_freq_offset = []
    for a in range(num_ant):
        est_timing.append(
            estimate_timing_offset(rx_symbols[a], pilot_symbols, pilot_indices, cfg)
        )
        est_freq_offset.append(
            estimate_frequency_offset(rx_symbols[a], pilot_symbols, pilot_indices, cfg)
        )
    if cfg.num_tx_ant == 1:
        est_time = np.array(est_timing)[:,None]
        est_freq = np.array(est_freq_offset)[:,None]
    else:
        est_time = np.array(est_timing)
        est_freq = np.array(est_freq_offset)
    est_timing = np.mean(est_time,axis=1)
    est_freq_offset = np.mean(est_freq, axis=1)

    # 逐层逐天线补偿频偏
    comp_ant = [
        compensate_frequency_offset(
            signal[a],
            est_freq_offset[a],
            cfg,
        )
        for a in range(num_ant)
    ]
    signal_freq_comp = np.stack(comp_ant, axis=0)

    # CP 移除与FFT
    rx_symbols = remove_cp_and_fft(signal_freq_comp, cfg)

    # 逐层逐天线补偿时延
    comp_ant = [
        compensate_timing_offset(
            rx_symbols[a],
            est_timing[a],
            cfg,
        )
        for a in range(num_ant)
    ]
    signal_timing = np.stack(comp_ant, axis=0) # (num_ant, num_symbols, n_subcarrier)
    # 3. 信道估计和均衡
    num_layer = cfg.num_tx_ant
    h_est_ant = []
    for a in range(num_ant):
        h_est_tmp = estimate_channel(signal_timing[a], cfg)
        h_est_ant.append(h_est_tmp)
    h_est = np.stack(h_est_ant, axis=0)
    # complex signal power normalize
    h_Power_scale = np.mean(np.abs(h_est)**2) 
    #print(f'h_Power_scale:',h_Power_scale)
    h_est = h_est / np.sqrt(h_Power_scale)
    signal_Power_scale = np.mean(np.abs(signal_timing)**2) 
    #print(f'signal_Power_scale:',signal_Power_scale)
    signal_timing = signal_timing / np.sqrt(signal_Power_scale)
    rx_real_part = np.real(signal_timing)
    rx_imag_part = np.imag(signal_timing)
    h_real_part = np.real(h_est)
    h_imag_part = np.imag(h_est)
    pilot_Power_scale = np.mean(np.abs(pilot_symbols_include_zeros[...,::2])**2)
    #print(pilot_Power_scale)
    pilot_symbols_include_zeros = pilot_symbols_include_zeros / np.sqrt(pilot_Power_scale)
    pilot_real_prat = np.real(pilot_symbols_include_zeros)
    pilot_imag_part = np.imag(pilot_symbols_include_zeros)    
    pilots_indx = cfg.get_pilot_symbol_indices()
    data_indx = cfg.get_data_symbol_indices()
    #导频数据的非导频位置信息置0
    dim1,dim2,dim3 = signal_timing.shape
    mask = np.zeros(dim3, dtype=bool)
    mask[pilot_indices] = True
    mask_3d = mask[np.newaxis, np.newaxis, :]  # 形状: (1, 1, dim3)
    pilot_stream = np.concatenate((rx_real_part[...,pilots_indx,::2], rx_imag_part[...,pilots_indx,::2],
                                   h_real_part[...,pilots_indx,::2], h_imag_part[...,pilots_indx,::2],
                                   pilot_real_prat[...,::2], pilot_imag_part[...,::2]), axis=0, dtype=np.float32)
    #print(pilot_stream.shape)                               
    pilot_stream = rb_view(torch.from_numpy(pilot_stream[np.newaxis,...]),1)
    #print(pilot_stream.shape)
    data_stream = np.concatenate((rx_real_part[...,data_indx,:], rx_imag_part[...,data_indx,:],
                                  h_real_part[...,data_indx,:], h_imag_part[...,data_indx,:]), axis=0, dtype=np.float32)  # 将实部和虚部堆叠成最后一维
    data_stream = rb_view(torch.from_numpy(data_stream[np.newaxis,...]))
    # print(data_stream.shape)
    if not run_onnx_flag:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if plus_falg:
            model_cfg = ModelCPlusConfig(
                d_in_pilot=10, d_in_data=8, d_model=128, d_model_data=128, nhead=8, nhead_data=8,
                num_enc_layers=4, num_dec_layers=8,
                dim_ff=512,dim_ff_data=512, dropout=0.1, bits_per_symbol=6
            )
            model = ModelCPlus(model_cfg).to(device)
        else:    
            model_cfg = ModelCConfig(
                d_in=8,
                d_model=64,
                nhead=4,
                num_enc_layers=2,
                num_dec_layers=3,
                dim_ff=128,
                dropout=0.1,
                bits_per_symbol=6,  # 16QAM
                )
            model = ModelC(model_cfg, n_sym=cfg.num_symbols).to(device)
        load_full(model, model_weights_path, device)
        with torch.no_grad():
            llr = model(pilot_stream, data_stream) #llr size is [batch, n_subcarriers, n_symbols, n_bits_per_symbol]
            if plus_falg:
                llr = llr["logits"]
        llr = llr[...,:cfg.mod_order]
        llr = rb_view_reverse(llr, 1)
    llr = llr.numpy()
    bits = np.squeeze((llr > 0).astype(np.int8))
    return bits

if __name__ == '__main__':
    # --- 使用示例 ---
    # 1. 创建一个OFDM配置
    config = OFDMConfig(
        n_fft=256,
        n_subcarrier=192,
        cp_len=16,
        mod_order=4,  # 16QAM
        num_symbols=14,
        num_tx_ant=1,
        num_rx_ant=2,
        snr_db=20,
        channel_type='multipath'
    )
    # 测试create_data
    gen = create_data(config, 1)
    for i in range(10):
        print(i)
        x_pilot,x_data,y_bits,z_llr = next(gen)
        print(x_pilot.shape)
        print(x_data.shape)
        print(y_bits.shape)
        print(z_llr.shape)
    # 2. 创建TensorFlow数据集
    BATCH_SIZE = 4
    STEPS_PER_EPOCH = 4
    train_dataset = create_pytorch_dataloader(config, BATCH_SIZE)

    # 3. 迭代数据集并检查数据形状
    for x_pilot_batch,x_data_batch, y_batch, z_batch in train_dataset:
        print("成功生成一批数据！")
        print(f"输入 (X)导频 形状: {x_pilot_batch.shape}")
        print(f"输入 (X)数据 形状: {x_data_batch.shape}")
        print(f"标签 (y) 形状: {y_batch.shape}")
        print(f"标签 (z) 形状: {z_batch.shape}")

    # 可以在这里将 `train_dataset` 传递给 `model.fit()`
    # model.fit(train_dataset, epochs=10)
    # model.fit(train_dataset, epochs=10)
