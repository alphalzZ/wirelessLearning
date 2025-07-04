"""
信道模型模块
作者：AI助手
日期：2024-05-31
"""

import numpy as np
import sys
from pathlib import Path
from numpy.typing import NDArray
from typing import Tuple, Optional

# 添加项目根目录到Python路径
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.config import OFDMConfig

def awgn_channel(signal: np.ndarray, num_rx: int = 1) -> np.ndarray:
    """AWGN信道

    Args:
        signal: 输入信号

    Returns:
        经过AWGN信道的信号
    """

    # 噪声功率固定为 1，发送端会根据 ``snr_db`` 缩放信号

    # 生成复高斯噪声 (E{|n|^2}=1)
    if num_rx == 1:
        noise_shape = signal.shape
    else:
        noise_shape = (num_rx, signal.shape[0]) if signal.ndim == 1 else signal.shape
    noise = (
        np.random.randn(*noise_shape) + 1j * np.random.randn(*noise_shape)
    ) / np.sqrt(2)

    if num_rx == 1:
        rx_signal = signal + noise
    else:
        if signal.ndim == 1:
            rx_signal = signal[None, :] + noise
        else:
            rx_signal = signal + noise

    return rx_signal


def rayleigh_channel(
    signal: NDArray[np.complex128],
    *,
    block_fading: bool = True,
    rng: Optional[np.random.Generator] = None,
    num_rx: int = 1,
) -> Tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """
    瑞利平坦衰落 (block 或 sample‑by‑sample) + AWGN

    参数
    ----
    signal        : (N,)  发送复基带
    block_fading  : True  -> 整帧 1 个系数; False -> 每采样独立
    rng           : numpy.random.Generator, 可选

    返回
    ----
    rx_signal     : (N,)  接收信号
    h             : (1,) 或 (N,) 信道复增益
    """
    if rng is None:
        rng = np.random.default_rng()

    # 1) 生成瑞利系数：CN(0,1) / sqrt(2)，功率均值 = 1
    if block_fading:
        h = (
            rng.standard_normal(num_rx) + 1j * rng.standard_normal(num_rx)
        ) / np.sqrt(2)
        h_sig = h[:, None]
    else:  # independent fast fading
        h = (
            rng.standard_normal((num_rx, signal.shape[0]))
            + 1j * rng.standard_normal((num_rx, signal.shape[0]))
        ) / np.sqrt(2)
        h_sig = h

    noise_shape = (num_rx, signal.shape[0]) if num_rx > 1 else signal.shape
    noise = (
        np.random.randn(*noise_shape) + 1j * np.random.randn(*noise_shape)
    ) / np.sqrt(2)

    # 3) 通过信道 + 加噪
    if num_rx == 1:
        rx_signal = h_sig * signal + noise
    else:
        rx_signal = h_sig * signal[None, :] + noise

    # reshape h 为 (N,) 方便后续逐样本处理；block_fading 时重复填充
    if block_fading:
        h = np.repeat(h_sig, signal.shape[0], axis=1)

    return rx_signal.astype(np.complex128), h.astype(np.complex128)


def multipath_channel(
    signal: NDArray[np.complex128],
    num_paths: int = 8,
    max_delay: int = 16,
    rng: Optional[np.random.Generator] = None,
    num_rx: int = 1,
) -> Tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    r"""
    频率选择性瑞利多径信道 + AWGN

    参数
    ----
    signal     : (N,)  发送时域复基带
    num_paths  : 多径条数 (含直射 LOS / 最早径)
    max_delay  : 最大离散时延 (采样数, 含 0)
    rng        : numpy.random.Generator, 便于复现 (默认全局 RNG)

    返回
    ----
    rx_signal  : shape 与 signal 相同, 经过信道 + 加噪
    h          : (max_delay+1,) 离散时域脉冲响应
    """
    if rng is None:
        rng = np.random.default_rng()

    # -------------------------------------------------
    # 1) 随机时延 —— 保证“互不重叠”且包含 0
    # -------------------------------------------------
    if max_delay < num_paths - 1:
        raise ValueError("max_delay 必须 ≥ num_paths-1")
    # choice 不放回抽取时延，含 0
    delays = rng.choice(np.arange(max_delay + 1), size=num_paths, replace=False)
    delays[0] = 0                           # 保证首条为直射径
    delays.sort()                           # 递增方便阅读

    # -------------------------------------------------
    # 2) 瑞利增益，单位平均功率
    # -------------------------------------------------
    gains = (rng.standard_normal(num_paths) + 1j * rng.standard_normal(num_paths)) / np.sqrt(2)
    gains /= np.sqrt(np.sum(np.abs(gains) ** 2))        # 归一化 ∑|g|² = 1

    # -------------------------------------------------
    # 3) 构造离散脉冲响应 h[n]
    # -------------------------------------------------
    h = np.zeros(max_delay + 1, dtype=np.complex128)
    h[delays] = gains                                   # 重叠时延已避免，无需累加

    # -------------------------------------------------
    # 4) 通过信道 (线性卷积) —— 长信号用 FFT 卷积更快
    # -------------------------------------------------
    rx_wo_noise_single = np.convolve(signal, h, mode="same")   # 保持与原长一致
    if num_rx == 1:
        rx_wo_noise = rx_wo_noise_single
    else:
        rx_wo_noise = np.tile(rx_wo_noise_single[None, :], (num_rx, 1))

    # -------------------------------------------------
    # 5) AWGN 噪声，参照接收端功率设 SNR
    # -------------------------------------------------

    noise_shape = (num_rx, signal.shape[0]) if num_rx > 1 else signal.shape
    noise = (
        np.random.randn(*noise_shape) + 1j * np.random.randn(*noise_shape)
    ) / np.sqrt(2)

    rx_signal = rx_wo_noise + noise
    return rx_signal, h


def sionna_fading_channel(
    signal: NDArray[np.complex128],
    *,
    block_fading: bool = True,
    num_rx: int = 1,
):
    """Rayleigh衰落信道封装，优先使用Sionna实现"""
    try:
        import tensorflow as tf
        from sionna.phy.channel import FlatFadingChannel
    except Exception as exc:  # pragma: no cover - optional dependency
        # 回退到本地实现
        return rayleigh_channel(signal, block_fading=block_fading, num_rx=num_rx)

    signal_tf = tf.constant(signal[None, :, None], dtype=tf.complex64)
    ch = FlatFadingChannel(
        num_tx_ant=1,
        num_rx_ant=num_rx,
        add_awgn=False
    )
    rx_tf = ch(signal_tf)
    rx_np = tf.squeeze(rx_tf).numpy().transpose()
    rx_np = awgn_channel(rx_np, num_rx=num_rx)
    return rx_np


def sionna_tdl_channel(
    signal: NDArray[np.complex128],
    *,
    model: str = "C",
    delay_spread: float = 300e-9,
    carrier_freq: float = 3.5e9,
    num_rx: int = 1,
) -> NDArray[np.complex128]:
    """3GPP TDL信道封装，优先使用Sionna实现"""
    try:
        from sionna.phy.channel.tr38901 import TDL
    except Exception as exc:  # pragma: no cover - optional dependency
        return multipath_channel(signal, num_rx=num_rx)[0]

    tdl = TDL(model, delay_spread, carrier_freq, num_rx_ant=num_rx, num_tx_ant=1)
    delays = tdl.delays.numpy()
    powers = tdl.mean_powers.numpy()

    # 简易基于平均功率的瑞利多径模型实现
    max_delay = int(np.ceil(len(delays)))
    delay_samples = np.round(delays / delays.max() * max_delay).astype(int)

    rng = np.random.default_rng()
    h = np.zeros((num_rx, delay_samples.max() + 1), dtype=np.complex128)
    for d, p in zip(delay_samples, powers):
        gain = (rng.standard_normal(num_rx) + 1j * rng.standard_normal(num_rx)) / np.sqrt(2)
        h[:, d] += gain * np.sqrt(p)

    if num_rx == 1:
        rx = np.convolve(signal, h[0], mode="same")
    else:
        rx = np.stack([np.convolve(signal, h[i], mode="same") for i in range(num_rx)], axis=0)

    rx = awgn_channel(rx, num_rx=num_rx)
    return rx

if __name__ == "__main__":
    # 创建测试配置
    cfg = OFDMConfig(
        n_fft=64,
        n_subcarrier=32,
        cp_len=16,
        mod_order=4,  # 16QAM
        num_symbols=10,  # 测试用较少的符号数
        pilot_symbols=[2,5],
        pilot_pattern='comb',
        pilot_spacing=2,
        est_method='linear',
        interp_method='linear',
        equalizer='zf',
        est_time='fft_ml',
        channel_type='awgn',
        display_est_result=False,
        code_rate=0.5,
    )
    
    # 生成测试信号
    np.random.seed(42)
    test_signal = (np.random.randn(1000) + 1j * np.random.randn(1000)) / np.sqrt(2)
    test_signal = test_signal * 10**(cfg.snr_db/20)
    
    # 测试AWGN信道
    print("测试AWGN信道...")
    rx_awgn = awgn_channel(test_signal)
    measured_snr = 10 * np.log10(np.mean(np.abs(test_signal)**2) / 
                                np.mean(np.abs(rx_awgn - test_signal)**2))
    print(f"目标SNR: {cfg.snr_db} dB")
    print(f"测量SNR: {measured_snr:.2f} dB")
    
    # 测试瑞利信道
    print("\n测试瑞利信道...")
    rx_rayleigh, h_rayleigh = rayleigh_channel(test_signal)
    print(f"信道响应形状: {h_rayleigh.shape}")
    print(f"信道响应功率: {np.mean(np.abs(h_rayleigh)**2):.3f}")
    
    # 测试多径信道
    print("\n测试多径信道...")
    rx_multipath, h_multipath = multipath_channel(test_signal)
    print(f"多径信道响应形状: {h_multipath.shape}")
    print(f"多径信道响应功率: {np.mean(np.abs(h_multipath)**2):.3f}") 