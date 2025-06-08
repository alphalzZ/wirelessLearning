"""
信道模型模块
作者：AI助手
日期：2024-05-31
"""

import numpy as np
from typing import Tuple
import sys
from pathlib import Path

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

def multipath_channel(signal: np.ndarray, snr_db: float, 
                     num_paths: int = 4, 
                     max_delay: int = 16) -> Tuple[np.ndarray, np.ndarray]:
    """多径衰落信道
    
    Args:
        signal: 输入信号
        snr_db: 信噪比(dB)
        num_paths: 多径数量
        max_delay: 最大时延（采样点数）
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: (经过多径信道的信号, 信道响应)
    """
    # 生成随机时延
    delays = np.random.randint(0, max_delay, num_paths)
    delays[0] = 0  # 第一条路径无时延
    
    # 生成随机复增益
    gains = (np.random.randn(num_paths) + 1j * np.random.randn(num_paths)) / np.sqrt(2)
    gains = gains / np.sqrt(np.sum(np.abs(gains)**2))  # 功率归一化
    
    # 构建信道响应
    h = np.zeros(max_delay, dtype=np.complex64)
    for delay, gain in zip(delays, gains):
        h[delay] = gain
    
    # 计算信号功率
    signal_power = np.mean(np.abs(signal) ** 2)
    
    # 计算噪声功率
    noise_power = signal_power / (10 ** (snr_db / 10))
    
    # 生成复高斯噪声
    noise = np.sqrt(noise_power/2) * (np.random.randn(*signal.shape) + 
                                     1j * np.random.randn(*signal.shape))
    
    # 通过信道并添加噪声
    rx_signal = np.convolve(signal, h, mode='same') + noise
    
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