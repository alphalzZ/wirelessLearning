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

def awgn_channel(signal: np.ndarray, snr_db: float) -> np.ndarray:
    """AWGN信道
    
    Args:
        signal: 输入信号
        snr_db: 信噪比(dB)
    
    Returns:
        经过AWGN信道的信号
    """
    # 计算信号功率
    signal_power = np.mean(np.abs(signal) ** 2)
    
    # 计算噪声功率
    noise_power = signal_power / (10 ** (snr_db / 10))
    
    # 生成复高斯噪声
    noise = np.sqrt(noise_power/2) * (np.random.randn(*signal.shape) + 
                                     1j * np.random.randn(*signal.shape))
    
    # 添加噪声
    rx_signal = signal + noise
    
    return rx_signal

def rayleigh_channel(signal: np.ndarray, snr_db: float) -> Tuple[np.ndarray, np.ndarray]:
    """瑞利衰落信道
    
    Args:
        signal: 输入信号
        snr_db: 信噪比(dB)
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: (经过瑞利信道的信号, 信道响应)
    """
    # 生成瑞利衰落信道响应
    h = (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape)) / np.sqrt(2)
    
    # 计算信号功率
    signal_power = np.mean(np.abs(signal) ** 2)
    
    # 计算噪声功率
    noise_power = signal_power / (10 ** (snr_db / 10))
    
    # 生成复高斯噪声
    noise = np.sqrt(noise_power/2) * (np.random.randn(*signal.shape) + 
                                     1j * np.random.randn(*signal.shape))
    
    # 通过信道并添加噪声
    rx_signal = h * signal + noise
    
    return rx_signal, h

def multipath_channel(
    signal: NDArray[np.complex128],
    snr_db: float,
    num_paths: int = 4,
    max_delay: int = 16,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    r"""
    频率选择性瑞利多径信道 + AWGN

    参数
    ----
    signal     : (N,)  发送时域复基带
    snr_db     : 目标平均 SNR (以 **接收端见到的符号功率** 为 1)
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
    rx_wo_noise = np.convolve(signal, h, mode="same")   # 保持与原长一致

    # -------------------------------------------------
    # 5) AWGN 噪声，参照接收端功率设 SNR
    # -------------------------------------------------
    rx_power = np.mean(np.abs(rx_wo_noise) ** 2)        # ≈1 (因我们已归一化)
    snr_lin  = 10 ** (snr_db / 10)
    noise_var = rx_power / snr_lin
    noise_std = np.sqrt(noise_var / 2)                  # 单独对实/虚分量

    noise = noise_std * (rng.standard_normal(signal.shape) +
                         1j * rng.standard_normal(signal.shape))

    rx_signal = rx_wo_noise + noise
    return rx_signal, h

if __name__ == "__main__":
    # 创建测试配置
    cfg = OFDMConfig(
        n_fft=64,
        cp_len=16,
        mod_order=4,  # 16QAM
        num_symbols=10  # 测试用较少的符号数
    )
    
    # 生成测试信号
    np.random.seed(42)
    test_signal = np.random.randn(1000) + 1j * np.random.randn(1000)
    test_signal = test_signal / np.sqrt(np.mean(np.abs(test_signal)**2))  # 功率归一化
    
    # 测试AWGN信道
    print("测试AWGN信道...")
    snr_db = 10
    rx_awgn = awgn_channel(test_signal, snr_db)
    measured_snr = 10 * np.log10(np.mean(np.abs(test_signal)**2) / 
                                np.mean(np.abs(rx_awgn - test_signal)**2))
    print(f"目标SNR: {snr_db} dB")
    print(f"测量SNR: {measured_snr:.2f} dB")
    
    # 测试瑞利信道
    print("\n测试瑞利信道...")
    rx_rayleigh, h_rayleigh = rayleigh_channel(test_signal, snr_db)
    print(f"信道响应形状: {h_rayleigh.shape}")
    print(f"信道响应功率: {np.mean(np.abs(h_rayleigh)**2):.3f}")
    
    # 测试多径信道
    print("\n测试多径信道...")
    rx_multipath, h_multipath = multipath_channel(test_signal, snr_db)
    print(f"多径信道响应形状: {h_multipath.shape}")
    print(f"多径信道响应功率: {np.mean(np.abs(h_multipath)**2):.3f}") 