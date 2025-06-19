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


def awgn_channel(
    signal: np.ndarray,
    *,
    num_rx: int = 1,
    num_tx: int = 1,
) -> np.ndarray:
    """AWGN信道

    Args:
        signal: (num_tx, N) 或 (N,) 输入信号
        num_rx: 接收天线数
        num_tx: 发送天线数

    Returns:
        (num_rx, N) 或 (N,) 加噪后信号
    """

    # 噪声功率固定为 1，发送端会根据 ``snr_db`` 缩放信号

    # 输入信号形状统一为 (num_tx, N)
    if signal.ndim == 1:
        signal = signal[None, :]
    N = signal.shape[1]
    num_tx = signal.shape[0]

    # 噪声功率固定为 1，发送端会根据 ``snr_db`` 缩放信号
    noise = (
        np.random.randn(num_rx, N) + 1j * np.random.randn(num_rx, N)
    ) / np.sqrt(2)

    # 恒等信道矩阵，将每个发送天线映射到对应接收天线
    H = np.eye(num_rx, num_tx, dtype=np.complex128)

    rx_signal = H @ signal + noise

    # 单天线场景保持 (N,) 输出
    if rx_signal.shape[0] == 1:
        rx_signal = rx_signal.reshape(-1)

    return rx_signal


def rayleigh_channel(
    signal: NDArray[np.complex128],
    *,
    block_fading: bool = True,
    rng: Optional[np.random.Generator] = None,
    num_rx: int = 1,
    num_tx: int = 1,
) -> Tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """
    瑞利平坦衰落 (block 或 sample‑by‑sample) + AWGN

    参数
    ----
    signal        : (num_tx, N) 或 (N,) 发送复基带
    block_fading  : True -> 整帧 1 个系数; False -> 每采样独立
    rng           : numpy.random.Generator, 可选
    num_rx        : 接收天线数
    num_tx        : 发送天线数

    返回
    ----
    rx_signal     : (num_rx, N) 接收信号
    h             : (num_rx, num_tx) 或 (num_rx, num_tx, N) 信道增益
    """
    if rng is None:
        rng = np.random.default_rng()

    if signal.ndim == 1:
        signal = signal[None, :]
    N = signal.shape[1]
    num_tx = signal.shape[0]

    # 1) 生成瑞利系数：CN(0,1) / sqrt(2)，功率均值 = 1
    if block_fading:
        H = (
            rng.standard_normal((num_rx, num_tx))
            + 1j * rng.standard_normal((num_rx, num_tx))
        ) / np.sqrt(2)
        H_time = H[:, :, None]
    else:  # independent fast fading
        H = (
            rng.standard_normal((num_rx, num_tx, N))
            + 1j * rng.standard_normal((num_rx, num_tx, N))
        ) / np.sqrt(2)
        H_time = H

    noise = (
        np.random.randn(num_rx, N) + 1j * np.random.randn(num_rx, N)
    ) / np.sqrt(2)

    rx_wo_noise = np.sum(H_time * signal[None, :, :], axis=1)
    rx_signal = rx_wo_noise + noise

    if block_fading:
        return rx_signal.astype(np.complex128), H.astype(np.complex128)
    else:
        return rx_signal.astype(np.complex128), H.astype(np.complex128)


def multipath_channel(
    signal: NDArray[np.complex128],
    num_paths: int = 8,
    max_delay: int = 16,
    rng: Optional[np.random.Generator] = None,
    num_rx: int = 1,
    num_tx: int = 1,
) -> Tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    r"""
    频率选择性瑞利多径信道 + AWGN

    参数
    ----
    signal     : (num_tx, N) 或 (N,) 发送时域复基带
    num_paths  : 多径条数 (含直射 LOS / 最早径)
    max_delay  : 最大离散时延 (采样数, 含 0)
    rng        : numpy.random.Generator, 便于复现 (默认全局 RNG)
    num_rx     : 接收天线数
    num_tx     : 发送天线数

    返回
    ----
    rx_signal  : (num_rx, N) 经过信道 + 加噪后的信号
    h          : (num_rx, num_tx, max_delay+1) 离散时域脉冲响应
    """
    if rng is None:
        rng = np.random.default_rng()

    if signal.ndim == 1:
        signal = signal[None, :]
    N = signal.shape[1]
    num_tx = signal.shape[0]

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
    gains = (
        rng.standard_normal((num_rx, num_tx, num_paths))
        + 1j * rng.standard_normal((num_rx, num_tx, num_paths))
    ) / np.sqrt(2)
    gains /= np.sqrt(np.sum(np.abs(gains) ** 2, axis=2, keepdims=True))

    # -------------------------------------------------
    # 3) 构造离散脉冲响应 h[n]
    # -------------------------------------------------
    h = np.zeros((num_rx, num_tx, max_delay + 1), dtype=np.complex128)
    h[:, :, delays] = gains  # shape (num_rx,num_tx,max_delay+1)

    # -------------------------------------------------
    # 4) 通过信道 (线性卷积) —— 长信号用 FFT 卷积更快
    # -------------------------------------------------
    rx_wo_noise = np.zeros((num_rx, N), dtype=np.complex128)
    for i in range(num_rx):
        for j in range(num_tx):
            rx_wo_noise[i] += np.convolve(signal[j], h[i, j], mode="same")

    # -------------------------------------------------
    # 5) AWGN 噪声，参照接收端功率设 SNR
    # -------------------------------------------------

    noise = (
        np.random.randn(num_rx, N) + 1j * np.random.randn(num_rx, N)
    ) / np.sqrt(2)

    rx_signal = rx_wo_noise + noise
    return rx_signal, h


def sionna_fading_channel(
    signal: NDArray[np.complex128],
    *,
    block_fading: bool = True,
    num_rx: int = 1,
    num_tx: int = 1,
):
    """Rayleigh衰落信道封装，优先使用Sionna实现

    参数
    ----
    signal       : (num_tx, N) 或 (N,) 输入信号
    block_fading : 是否块衰落
    num_rx       : 接收天线数
    num_tx       : 发送天线数

    返回
    ----
    接收信号 ``(num_rx, N)``
    """
    try:
        import tensorflow as tf
        from sionna.phy.channel import FlatFadingChannel
    except Exception as exc:  # pragma: no cover - optional dependency
        # 回退到本地实现，仅返回接收信号
        rx, _ = rayleigh_channel(
            signal,
            block_fading=block_fading,
            num_rx=num_rx,
            num_tx=num_tx,
        )
        return rx

    if signal.ndim == 1:
        signal = signal[None, :]
    signal_tf = tf.constant(signal.transpose(1, 0)[None, :, :], dtype=tf.complex64)
    ch = FlatFadingChannel(
        num_tx_ant=num_tx,
        num_rx_ant=num_rx,
        add_awgn=False
    )
    rx_tf = ch(signal_tf)
    rx_np = tf.transpose(tf.squeeze(rx_tf), perm=[1, 0]).numpy()
    rx_np = awgn_channel(rx_np, num_rx=num_rx, num_tx=num_tx)
    return rx_np


def sionna_tdl_channel(
    signal: NDArray[np.complex128],
    *,
    model: str = "C",
    delay_spread: float = 300e-9,
    carrier_freq: float = 3.5e9,
    num_rx: int = 1,
    num_tx: int = 1,
) -> Tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """3GPP TDL信道封装，优先使用Sionna实现

    参数
    ----
    signal        : (num_tx, N) 或 (N,) 输入信号
    model         : TDL模型类别
    delay_spread  : 时延扩展
    carrier_freq  : 载波频率
    num_rx        : 接收天线数
    num_tx        : 发送天线数

    返回
    ----
    rx_signal : (num_rx, N) 接收信号
    h         : (num_rx, num_tx, L) 信道脉冲响应
    """
    try:
        from sionna.phy.channel.tr38901 import TDL
    except Exception as exc:  # pragma: no cover - optional dependency
        return multipath_channel(
            signal,
            num_rx=num_rx,
            num_tx=num_tx,
        )

    tdl = TDL(model, delay_spread, carrier_freq, num_rx_ant=num_rx, num_tx_ant=num_tx)
    delays = tdl.delays.numpy()
    powers = tdl.mean_powers.numpy()

    # 简易基于平均功率的瑞利多径模型实现
    max_delay = int(np.ceil(len(delays)))
    delay_samples = np.round(delays / delays.max() * max_delay).astype(int)

    rng = np.random.default_rng()
    h = np.zeros((num_rx, num_tx, delay_samples.max() + 1), dtype=np.complex128)
    for d, p in zip(delay_samples, powers):
        gain = (
            rng.standard_normal((num_rx, num_tx))
            + 1j * rng.standard_normal((num_rx, num_tx))
        ) / np.sqrt(2)
        h[:, :, d] += gain * np.sqrt(p)

    if signal.ndim == 1:
        signal = signal[None, :]
    N = signal.shape[1]
    rx = np.zeros((num_rx, N), dtype=np.complex128)
    for i in range(num_rx):
        for j in range(num_tx):
            rx[i] += np.convolve(signal[j], h[i, j], mode="same")

    rx = awgn_channel(rx, num_rx=num_rx, num_tx=num_tx)
    return rx, h

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
