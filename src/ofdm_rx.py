"""
OFDM接收端处理模块
作者：AI助手
日期：2024-05-31
"""

import numpy as np
from typing import Tuple, Optional
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from numpy.typing import NDArray

plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows 黑体
# plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # Windows 微软雅黑
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # macOS
# plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC']  # Linux

plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# 添加项目根目录到Python路径
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.config import OFDMConfig

# def estimate_frequency_offset(rx_symbols: np.ndarray, pilot_symbols: np.ndarray, 
#                             pilot_indices: np.ndarray, cfg: OFDMConfig) -> float:
#     """
#     基于两帧导频的相位差（arctan公式）估计频偏
#     """
#     pilot_symbol_indices = cfg.get_pilot_symbol_indices()
#     if len(pilot_symbol_indices) < 2:
#         raise ValueError("至少需要两个含导频的OFDM符号用于频偏估计")
#     idx1, idx2 = pilot_symbol_indices[:2]
#     delta_t = idx2 - idx1

#     # 标准信道补偿
#     Y1 = rx_symbols[idx1, pilot_indices] / pilot_symbols
#     Y2 = rx_symbols[idx2, pilot_indices] / pilot_symbols

#     prod = np.conj(Y1) * Y2
#     num = np.sum(np.imag(prod))
#     den = np.sum(np.real(prod))
#     eps_hat = (1 / (2 * np.pi * delta_t)) * np.arctan2(num, den)
#     return eps_hat

def estimate_frequency_offset(rx_symbols: np.ndarray, pilot_symbols: np.ndarray, 
                            pilot_indices: np.ndarray, cfg: OFDMConfig) -> float:
    """
    基于两帧导频的相位差估计频偏（支持非相邻导频符号）
    """
    pilot_symbol_indices = cfg.get_pilot_symbol_indices()
    if len(pilot_symbol_indices) < 2:
        raise ValueError("至少需要两个含导频的OFDM符号用于频偏估计")
    
    # 选择前两个导频符号索引
    idx1, idx2 = pilot_symbol_indices[:2]
    
    # 关键修复：计算实际符号间隔（考虑非相邻情况）
    delta_symbols = idx2 - idx1  # 符号间隔数
    
    # 计算实际时间间隔（考虑CP长度）
    symbol_duration = cfg.n_fft + cfg.cp_len  # 每个符号的总样本数（包括CP）
    delta_t = delta_symbols * symbol_duration / cfg.n_fft  # 归一化的时间间隔
    
    # 提取导频数据
    rx_pilot1 = rx_symbols[idx1, pilot_indices]
    rx_pilot2 = rx_symbols[idx2, pilot_indices]
    
    # 信道补偿
    Y1 = rx_pilot1 / pilot_symbols
    Y2 = rx_pilot2 / pilot_symbols
    
    # 计算相位差
    prod = np.conj(Y1) * Y2
    num = np.sum(np.imag(prod))
    den = np.sum(np.real(prod))
    
    # 估计频偏
    eps_hat = (1 / (2 * np.pi * delta_t)) * np.arctan2(num, den)
    return eps_hat

def compensate_frequency_offset(
    signal: NDArray[np.complex128],
    freq_offset: float,
    cfg: "OFDMConfig",
) -> NDArray[np.complex128]:
    """
    频偏补偿

    两种场景：
    1. `signal.ndim == 1` —— 时域序列（含或不含 CP），对每个采样直接旋转；
    2. `signal.ndim == 2` —— 去 CP + FFT 后的 **频域帧矩阵**，
       视 CFO 仅造成"公共相位误差 (CPE)"，对每个 OFDM 符号整行旋转。

    参数
    ----
    signal      : ndarray
        • 时域: shape (Ns,)                    — 复基带采样  
        • 频域: shape (Nsym, Nfft)             — 每行 1 个 OFDM 符号
    freq_offset : float
        归一化 CFO (单位：子载波间隔，范围约 -0.5 ~ 0.5)  
        正值表示接收端载波 **低** 于发送端
    cfg         : OFDMConfig
        提供 `n_fft`, `cp_len`

    返回
    ----
    compensated : ndarray，与输入形状一致
    """
    if signal.ndim == 1:  # ------- 时域补偿 -------
        t = np.arange(signal.size)                       # 采样索引
        phase = np.exp(-1j * 2 * np.pi * freq_offset * t / cfg.n_fft)
        return signal * phase.astype(signal.dtype)

    elif signal.ndim == 2:  # ----- 频域：仅补公共相位 -----
        n_sym, n_fft = signal.shape
        if n_fft != cfg.n_fft:
            raise ValueError("signal.shape[1] 与 cfg.n_fft 不一致")

        # 每个 OFDM 符号对应的时域"参考采样"索引:
        # 起始点位于 CP 尾端 → n0 = Ncp + m·(N+Ncp)
        sample_idx = cfg.cp_len + np.arange(n_sym) * (cfg.n_fft + cfg.cp_len)

        # CPE 补偿因子 e^{-j2π ε n0/N}
        cpe = np.exp(-1j * 2 * np.pi * freq_offset * sample_idx / cfg.n_fft).astype(
            signal.dtype
        )

        # 广播乘，shape (Nsym, 1) × (Nsym, Nfft) → (Nsym, Nfft)
        return signal * cpe[:, None]

    else:
        raise ValueError("signal 维度既不是 1 也不是 2，无法补偿")

def estimate_timing_offset(rx_symbols: np.ndarray, pilot_symbols: np.ndarray, 
                         pilot_indices: np.ndarray, cfg: OFDMConfig) -> int:
    """使用导频符号估计符号定时偏移
    
    Args:
        rx_symbols: 接收符号
        pilot_symbols: 导频符号
        pilot_indices: 导频位置
        cfg: 系统配置参数
    
    Returns:
        估计的定时偏移（采样点数）
    """
    # 使用接收信号的导频位置与本地导频计算信道响应
    # rx_pilots = rx_symbols[cfg.pilot_symbols, pilot_indices]
    # 创建行索引和列索引的网格
    rx_pilots = np.zeros((len(cfg.pilot_symbols), len(pilot_indices)), dtype=rx_symbols.dtype)

    for i, sym_idx in enumerate(cfg.pilot_symbols):
        rx_pilots[i, :] = rx_symbols[sym_idx, pilot_indices]

    # 使用广播机制提取数据
    h_pilot = rx_pilots / pilot_symbols
        
    # 计算导频位置的相位
    phases = np.angle(h_pilot)
    
    # 计算相邻导频的相位差
    phase_diff = np.diff(phases)
    
    # 处理相位跳变
    phase_diff = np.unwrap(phase_diff)
    
    # 计算导频间隔
    pilot_spacing = np.diff(pilot_indices)
    
    # 使用最小二乘法估计相位斜率
    # 构建线性方程组：phase_diff = slope * pilot_spacing
    slope = np.mean(phase_diff ) / np.mean(pilot_spacing)
    
    # 将相位斜率转换为采样点偏移
    timing_offset = int(round(slope * cfg.n_fft / (2 * np.pi)))
    
    return timing_offset

def estimate_channel(rx_symbols: np.ndarray, cfg: OFDMConfig, pilot_symbols: np.ndarray = None, 
                    pilot_indices: np.ndarray = None) -> np.ndarray:
    """信道估计
    
    Args:
        rx_symbols: 接收符号
        cfg: 系统配置参数
        pilot_symbols: 导频符号（可选）
        pilot_indices: 导频位置（可选）
    
    Returns:
        估计的信道响应
    """
    if pilot_symbols is None or pilot_indices is None:
        pilot_symbols = cfg.get_pilot_symbols()
        pilot_symbol_indices = cfg.get_pilot_symbol_indices()
        pilot_indices = cfg.get_pilot_indices()
    
    # 使用接收信号的导频位置与本地导频计算信道响应
    # 修改索引方式，确保维度匹配
    rx_pilots = np.zeros((len(pilot_symbol_indices), cfg.n_fft), dtype=np.complex64)
    for i, sym_idx in enumerate(pilot_symbol_indices):
        rx_pilots[i, :] = rx_symbols[sym_idx, :]
    
    #将rx_pilots和rx_symbols的维度对齐
    pilot_symbols_pad = np.zeros(cfg.n_fft, dtype=np.complex64)
    pilot_symbols_pad[pilot_indices] = pilot_symbols

    # 计算每个导频位置的信道响应，使用共轭相乘
    h_pilot = rx_pilots * np.conj(pilot_symbols_pad)
    
    # 如果pilot_symbols_pad不为0，则将h_pilot的值除以pilot_symbols_pad模的平方
    for i in range(len(pilot_symbol_indices)):
        idx = np.abs(pilot_symbols_pad) != 0
        h_pilot[i, idx] = h_pilot[i, idx] / np.abs(pilot_symbols_pad[idx]) ** 2
    
    #非导频位置的信道响应为其相邻导频位置的信道响应的平均
    h_est = np.zeros((len(pilot_symbol_indices), cfg.n_fft), dtype=np.complex64)
    for i in range(cfg.n_fft):
        if i in pilot_indices:
            h_est[:, i] = h_pilot[:, i]
        else:
            # 找到最近的导频位置
            left_pilots = pilot_indices[pilot_indices < i]
            right_pilots = pilot_indices[pilot_indices > i]
            
            if len(left_pilots) > 0 and len(right_pilots) > 0:
                # 正常情况：左右都有导频
                left_pilot = left_pilots[-1]
                right_pilot = right_pilots[0]
                alpha = (i - left_pilot) / (right_pilot - left_pilot)
                h_est[:, i] = (1 - alpha) * h_pilot[:, left_pilot] + alpha * h_pilot[:, right_pilot]
            elif len(left_pilots) > 0:
                # 只有左边有导频（最后一个导频之后的位置）
                h_est[:, i] = h_pilot[:, left_pilots[-1]]
            else:
                # 只有右边有导频（第一个导频之前的位置）
                h_est[:, i] = h_pilot[:, right_pilots[0]]
    
    # 使用滑动平均计算每个窗口的信道响应，窗口大小为2*cfg.pilot_spacing+1
    window_size = 2 * cfg.pilot_spacing + 1
    h_est_smooth = np.zeros_like(h_est)
    for i in range(len(pilot_symbol_indices)):
        for j in range(cfg.n_fft):
            # 计算当前窗口的起始和结束位置
            start = max(0, j - cfg.pilot_spacing)
            end = min(cfg.n_fft, j + cfg.pilot_spacing + 1)
            # 计算实际窗口大小
            actual_window_size = end - start
            # 使用实际窗口大小进行平均
            h_est_smooth[i, j] = np.mean(h_est[i, start:end])
    
    # 对h_est在每个子载波维度进行插值
    h_est_interp = np.zeros((cfg.num_symbols, cfg.n_fft), dtype=np.complex64)
    for i in range(cfg.n_fft):
        # 分别对实部和虚部进行插值
        real_interp = np.interp(np.arange(cfg.num_symbols), 
                              pilot_symbol_indices, 
                              h_est_smooth[:, i].real,
                              left=h_est_smooth[0, i].real,
                              right=h_est_smooth[-1, i].real)
        
        imag_interp = np.interp(np.arange(cfg.num_symbols), 
                              pilot_symbol_indices, 
                              h_est_smooth[:, i].imag,
                              left=h_est_smooth[0, i].imag,
                              right=h_est_smooth[-1, i].imag)
        
        # 组合实部和虚部
        h_est_interp[:, i] = real_interp + 1j * imag_interp
    
    return h_est_interp

def noise_var_estimate(rx_symbols: np.ndarray, Hest: np.ndarray, cfg: OFDMConfig) -> float:
    """噪声方差估计
    使用真实发送的导频和用信道估计恢复的导频的残差估计噪声方差
    Args:
        rx_symbols: 接收符号
        Hest: 信道估计结果，包含所有ofdm符号的信道估计
        cfg: 系统配置参数
    Returns:
        float: 估计的噪声方差
    """
    pilot_symbols = cfg.get_pilot_symbols()
    pilot_indices = cfg.get_pilot_indices()
    pilot_symbols_indice = cfg.get_pilot_symbol_indices()
    
    # 使用循环方式计算每个导频符号的噪声方差
    total_noise = 0
    for i, sym_idx in enumerate(pilot_symbols_indice):
        # 获取当前导频符号位置上的接收符号
        rx_pilot = rx_symbols[sym_idx, pilot_indices]
        # 获取当前导频符号位置上的信道估计值
        h_est = Hest[sym_idx, pilot_indices]
        # 计算与真实导频符号的差异
        rx_pilot_est = h_est*pilot_symbols
        diff = rx_pilot - rx_pilot_est
        # 累加噪声方差
        total_noise += np.mean(np.abs(diff) ** 2)
    
    # 返回平均噪声方差
    return total_noise / len(pilot_symbols_indice)

def channel_equalization(
    rx_symbols: NDArray[np.complex128],
    h_est: NDArray[np.complex128],
    noise_var: Optional[float] = None,
) -> NDArray[np.complex128]:
    """
    MMSE/ZF 频域均衡器

    对每个子载波 k 执行
        X̂_k = H*_k · R_k / ( |H_k|² + σ² )
    当 noise_var=None 或 0 时退化为零强制 (ZF)。

    参数
    ----
    rx_symbols : shape (..., Nfft)
        接收频域符号（去 CP、FFT 后）
    h_est      : shape (..., Nfft) 或 (Nfft,)
        频域信道估计；可按符号、天线广播
    noise_var  : float, optional
        复 AWGN 单复采样方差 σ²；若为 None 则默认为 0 (ZF)

    返回
    ----
    eq_symbols : 与 rx_symbols 同形状
        均衡后的符号
    """
    if noise_var is None:
        noise_var = 0.0

    # 保证广播： h_est 可是一维 (Nfft,) 也可与 rx 符合
    # conj(H) / (|H|^2 + σ²)
    denom = np.abs(h_est) ** 2 + noise_var
    # 防止 0/0；若 H≈0, 用 epsilon 保底, 避免 nan/inf
    eps = np.finfo(rx_symbols.dtype).eps
    denom = np.maximum(denom, eps)

    w_mmse = np.conj(h_est) / denom
    eq_symbols = rx_symbols * w_mmse  # 广播乘

    return eq_symbols


def remove_cp_and_fft(signal: np.ndarray, cfg: OFDMConfig) -> np.ndarray:
    """移除循环前缀并进行FFT
    
    Args:
        signal: 接收到的时域信号
        cfg: 系统配置参数
    
    Returns:
        频域符号，形状为(num_symbols, n_fft)
    """
    # 计算符号长度（包括CP）
    symbol_len = cfg.n_fft + cfg.cp_len
    
    # 计算符号数量
    num_symbols = len(signal) // symbol_len
    
    # 初始化频域符号数组
    rx_symbols = np.zeros((num_symbols, cfg.n_fft), dtype=np.complex64)
    
    for i in range(num_symbols):
        # 提取当前符号（去除CP）
        start_idx = i * symbol_len + cfg.cp_len
        end_idx = start_idx + cfg.n_fft
        time_symbol = signal[start_idx:end_idx]
        
        # FFT
        rx_symbols[i] = np.fft.fft(time_symbol, cfg.n_fft)
    
    return rx_symbols

def ofdm_rx(signal: np.ndarray, cfg: OFDMConfig) -> np.ndarray:
    """OFDM接收端处理
    
    Args:
        signal: 接收到的时域信号
        cfg: 系统配置参数
        pilot_symbols: 导频符号（可选）
        pilot_indices: 导频位置（可选）
    
    Returns:
        解调后的比特流
    """
    # 1. 移除循环前缀并进行FFT
    rx_symbols = remove_cp_and_fft(signal, cfg)
    
    # 2. 使用导频进行频偏估计和补偿
    pilot_symbols = cfg.get_pilot_symbols()
    pilot_indices = cfg.get_pilot_indices()
    est_timing = estimate_timing_offset(rx_symbols, pilot_symbols, pilot_indices, cfg)
    # 时延补偿
    # signal_timing = np.roll(signal,est_timing)
    # rx_symbols = remove_cp_and_fft(signal_timing, cfg)
    signal_timing = np.zeros_like(rx_symbols)
    for i in range(signal_timing.shape[0]):
        # 补偿相位旋转
        phase_compensation = np.exp(-1j * 2 * np.pi *est_timing * np.arange(cfg.n_fft) / cfg.n_fft)
        signal_timing[i] = rx_symbols[i] * phase_compensation
    # 频偏补偿
    est_freq_offset = estimate_frequency_offset(signal_timing, pilot_symbols, pilot_indices, cfg)
    rx_symbols_freq_compensation = compensate_frequency_offset(signal_timing, est_freq_offset, cfg)
    # rx_symbols_freq_compensation = remove_cp_and_fft(rx_symbols_freq_compensation, cfg)
    # 3. 信道估计和均衡
    h_est = estimate_channel(rx_symbols_freq_compensation, cfg)
    noise_var = noise_var_estimate(rx_symbols_freq_compensation, h_est, cfg)
    print(f"估计的SINR: {10*np.log10(1/noise_var)-6 :.2f} dB")
    #信道均衡
    rx_symbols_equalized = channel_equalization(rx_symbols_freq_compensation, h_est, noise_var)
    
    # 4. 解调（这里需要实现QAM解调）
    # TODO: 实现QAM解调
    bits_rx = np.zeros(rx_symbols_equalized.shape[0] * cfg.n_fft * cfg.mod_order, dtype=np.int8)
    
    return rx_symbols_equalized

if __name__ == "__main__":
    # 创建测试配置
    cfg = OFDMConfig(
        n_fft=256,
        cp_len=16,
        mod_order=4,  # 16QAM
        num_symbols=14,  # 测试用较少的符号数
        pilot_pattern='comb',
        pilot_spacing=2,  # 导频间隔
        pilot_symbols=[1,5]  # 在第2和第11个符号上插入导频
    )
    
    # 生成随机比特流
    np.random.seed(42)
    total_bits = cfg.get_total_bits()
    bits = np.random.randint(0, 2, total_bits)
    
    # 使用ofdm_tx模块生成OFDM时域信号
    from src.ofdm_tx import ofdm_tx, insert_pilots
    
    # 生成OFDM符号
    time_signal, pilot_indices, data_indices, freq_symbols = ofdm_tx(bits, cfg)
    
    # 获取导频信息
    pilot_symbols = cfg.get_pilot_symbols()
    
    # 打印调试信息
    print(f"时域信号长度: {len(time_signal)}")
    print(f"导频位置数量: {len(pilot_indices)}")
    print(f"导频符号数量: {len(pilot_symbols)}")
    
    # 测试频偏估计和补偿
    print("\n测试频偏估计...")
    freq_offset = 0.1  # 添加频偏
    t = np.arange(len(time_signal))
    phase_rotation = 2 * np.pi * freq_offset * t / cfg.n_fft
    signal_with_freq_offset = time_signal * np.exp(1j * phase_rotation)
    
    # 移除CP并进行FFT
    rx_symbols_freq_offset = remove_cp_and_fft(signal_with_freq_offset, cfg)
    
    # 估计频偏
    est_freq_offset = estimate_frequency_offset(rx_symbols_freq_offset, pilot_symbols, pilot_indices, cfg)
    print(f"实际频偏: {freq_offset:.3f}")
    print(f"估计频偏: {est_freq_offset:.3f}")
    # 频域频偏补偿
    # ---------- 1. 构造 per‑subcarrier 相位斜率向量 ----------
    # N = cfg.n_fft
    # k = np.arange(N)                     # 0 .. N-1
    # # e^{-j pi ε (1 - 2k/N)}   =   e^{-j pi ε} · e^{+j 2π ε k / N}
    # slope = np.exp(+1j * 2 * np.pi * est_freq_offset * (k / N - 0.5))

    # # ---------- 2. 构造 per‑symbol CPE 向量 ----------
    # n_sym = rx_symbols_freq_offset.shape[0]
    # time_index = np.arange(n_sym) * (N + cfg.cp_len) + cfg.cp_len
    # cpe = np.exp(
    #     -1j * 2 * np.pi * est_freq_offset * time_index / N
    # ).astype(rx_symbols_freq_offset.dtype)
    # ---------- 3. 频域补偿：广播乘 ----------
    # rx_symbols_freq_compensation = rx_symbols_freq_offset * (cpe[:, None] * slope[None, :])

    # 时域频偏补偿
    # phase_compensation = np.exp(-1j * 2 * np.pi * est_freq_offset * t / cfg.n_fft)
    # signal_with_freq_offset_compensation = signal_with_freq_offset * phase_compensation
    signal_with_freq_offset_compensation = compensate_frequency_offset(signal_with_freq_offset, est_freq_offset, cfg)
    rx_symbols_freq_compensation = remove_cp_and_fft(signal_with_freq_offset_compensation, cfg)

    
    # 测试时延估计和补偿
    print("\n测试时延估计...")
    timing_offset = 10  # 添加时延
    signal_with_timing = np.roll(time_signal, timing_offset)
    
    # 移除CP并进行FFT
    rx_symbols_timing = remove_cp_and_fft(signal_with_timing, cfg)
    rx_symbols_timing_compensation = np.zeros_like(rx_symbols_timing)
    # 估计时延
    est_timing = estimate_timing_offset(rx_symbols_timing, pilot_symbols, pilot_indices, cfg)
    print(f"实际时延: {timing_offset}")
    print(f"估计时延: {est_timing}")
    # 时延补偿
    for i in range(rx_symbols_timing.shape[0]):
        # 补偿相位旋转
        phase_compensation = np.exp(-1j * 2 * np.pi *est_timing * np.arange(cfg.n_fft) / cfg.n_fft)
        rx_symbols_timing_compensation[i] = rx_symbols_timing[i] * phase_compensation
    
    # 获取含导频的OFDM符号索引和数据OFDM符号索引
    pilot_symbol_indices = cfg.get_pilot_symbol_indices()
    data_symbol_indices = [i for i in range(cfg.num_symbols) if not cfg.has_pilot(i) or i not in pilot_symbol_indices]
    # 若所有符号都含导频，则数据符号索引取第一个非导频符号，否则取第一个符号
    data_symbol_idx = data_symbol_indices[0] if data_symbol_indices else 0
    pilot_symbol_idx = pilot_symbol_indices[0]

    plt.figure(figsize=(15, 10))
    
    # 绘制频偏测试结果
    plt.subplot(231)
    plt.scatter(freq_symbols[data_symbol_idx].real, freq_symbols[data_symbol_idx].imag, c='blue', label='数据')
    plt.scatter(freq_symbols[pilot_symbol_idx, pilot_indices].real, freq_symbols[pilot_symbol_idx, pilot_indices].imag, 
               c='red', marker='x', label='导频')
    plt.grid(True)
    plt.axis('equal')
    plt.xlabel('实部')
    plt.ylabel('虚部')
    plt.title(f'原始OFDM符号（数据符号{data_symbol_idx}，导频符号{pilot_symbol_idx}）')
    plt.legend()
    
    plt.subplot(232)
    plt.scatter(rx_symbols_freq_offset[data_symbol_idx].real, rx_symbols_freq_offset[data_symbol_idx].imag, c='blue', label='数据')
    plt.scatter(rx_symbols_freq_offset[pilot_symbol_idx, pilot_indices].real, rx_symbols_freq_offset[pilot_symbol_idx, pilot_indices].imag, 
               c='red', marker='x', label='导频')
    plt.grid(True)
    plt.axis('equal')
    plt.xlabel('实部')
    plt.ylabel('虚部')
    plt.title(f'加频偏后的符号\n(数据符号{data_symbol_idx}，导频符号{pilot_symbol_idx}，频偏={freq_offset:.3f})')
    plt.legend()
    
    plt.subplot(233)
    plt.scatter(rx_symbols_freq_compensation[data_symbol_idx].real, rx_symbols_freq_compensation[data_symbol_idx].imag, c='blue', label='数据')
    plt.scatter(rx_symbols_freq_compensation[pilot_symbol_idx, pilot_indices].real, rx_symbols_freq_compensation[pilot_symbol_idx, pilot_indices].imag, 
               c='red', marker='x', label='导频')
    plt.grid(True)
    plt.axis('equal')
    plt.xlabel('实部')
    plt.ylabel('虚部')
    plt.title(f'频偏补偿后的符号\n(数据符号{data_symbol_idx}，导频符号{pilot_symbol_idx}，估计频偏={est_freq_offset:.3f})')
    plt.legend()
    
    # 绘制时延测试结果
    plt.subplot(234)
    plt.scatter(freq_symbols[data_symbol_idx].real, freq_symbols[data_symbol_idx].imag, c='blue', label='数据')
    plt.scatter(freq_symbols[pilot_symbol_idx, pilot_indices].real, freq_symbols[pilot_symbol_idx, pilot_indices].imag, 
               c='red', marker='x', label='导频')
    plt.grid(True)
    plt.axis('equal')
    plt.xlabel('实部')
    plt.ylabel('虚部')
    plt.title(f'原始OFDM符号（数据符号{data_symbol_idx}，导频符号{pilot_symbol_idx}）')
    plt.legend()
    
    plt.subplot(235)
    plt.scatter(rx_symbols_timing[data_symbol_idx].real, rx_symbols_timing[data_symbol_idx].imag, c='blue', label='数据')
    plt.scatter(rx_symbols_timing[pilot_symbol_idx, pilot_indices].real, rx_symbols_timing[pilot_symbol_idx, pilot_indices].imag, 
               c='red', marker='x', label='导频')
    plt.grid(True)
    plt.axis('equal')
    plt.xlabel('实部')
    plt.ylabel('虚部')
    plt.title(f'加时延后的符号\n(数据符号{data_symbol_idx}，导频符号{pilot_symbol_idx}，时延={timing_offset})')
    plt.legend()
    
    plt.subplot(236)
    plt.scatter(rx_symbols_timing_compensation[data_symbol_idx].real, rx_symbols_timing_compensation[data_symbol_idx].imag, c='blue', label='数据')
    plt.scatter(rx_symbols_timing_compensation[pilot_symbol_idx, pilot_indices].real, rx_symbols_timing_compensation[pilot_symbol_idx, pilot_indices].imag, 
               c='red', marker='x', label='导频')
    plt.grid(True)
    plt.axis('equal')
    plt.xlabel('实部')
    plt.ylabel('虚部')
    plt.title(f'时延补偿后的符号\n(数据符号{data_symbol_idx}，导频符号{pilot_symbol_idx}，估计时延={est_timing})')
    plt.legend()
    
    plt.tight_layout()
    plt.show() 