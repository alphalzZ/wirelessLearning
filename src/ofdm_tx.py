"""
OFDM发送端处理模块
作者：AI助手
日期：2024-05-31
"""

import numpy as np
from typing import Tuple, Optional
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows 黑体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # Windows 微软雅黑
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # macOS
# plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC']  # Linux

plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 添加项目根目录到Python路径
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.config import OFDMConfig

import numpy as np

def qam_modulation(bits: np.ndarray, Qm: int) -> np.ndarray:
    """
    5G?NR Gray?coded QAM (38.211?§5.1)
    Qm = 2 (QPSK) | 4 (16QAM) | 6 (64QAM)
    Returns power?normalized symbols (E{|d|^2}=1).
    """
    if Qm not in (2, 4, 6):
        raise ValueError("Qm must be 2, 4 or 6")
    if bits.size % Qm:
        raise ValueError(f"len(bits) must be a multiple of Qm={Qm}")

    b = bits.astype(np.int8).reshape(-1, Qm)   # 强制 0/1 整数

    if Qm == 2:                     # QPSK
        i = 1 - 2 * b[:, 0]
        q = 1 - 2 * b[:, 1]
        norm = np.sqrt(2)

    elif Qm == 4:                   # 16?QAM  (±1, ±3)
        i = (1 - 2 * b[:, 0]) * (1 + 2 * b[:, 1])
        q = (1 - 2 * b[:, 2]) * (1 + 2 * b[:, 3])
        norm = np.sqrt(10)

    else:                           # 64?QAM  (±1, ±3, ±5, ±7)
        i = (1 - 2 * b[:, 0]) * (1 + 2 * b[:, 1] + 4 * b[:, 2])
        q = (1 - 2 * b[:, 3]) * (1 + 2 * b[:, 4] + 4 * b[:, 5])
        norm = np.sqrt(42)

    return (i + 1j * q) / norm


def insert_pilots(cfg: OFDMConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """插入导频符号
    
    Args:
        data_symbols: 数据符号
        cfg: 系统配置参数
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (导频符号, 导频位置, 数据位置)
    """
    # 获取导频位置和数据位置
    pilot_indices = cfg.get_pilot_indices()
    
    # 生成导频符号
    pilot_symbols = cfg.get_pilot_symbols()
    
    # 创建完整的OFDM符号
    ofdm_symbol = np.zeros(cfg.n_fft, dtype=np.complex64)
    ofdm_symbol[pilot_indices] = pilot_symbols
    
    return ofdm_symbol

def ofdm_tx(bits: np.ndarray, cfg: OFDMConfig) -> Tuple[np.ndarray, np.ndarray]:
    """OFDM发送端处理
    
    Args:
        bits: 输入比特流
        cfg: 系统配置参数
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (时域信号, 频域符号)
    """
    # 验证输入
    total_bits = cfg.get_total_bits()
    if len(bits) != total_bits:
        raise ValueError(f"输入比特流长度必须是{total_bits}")
    
    # 计算每个OFDM符号的比特数
    bits_per_symbol = cfg.get_total_bits_per_symbol()
    
    # 初始化时域信号数组和频域符号数组
    time_signal = np.array([], dtype=np.complex64)
    freq_symbols = np.zeros((cfg.num_symbols, cfg.n_fft), dtype=np.complex64)
    carrier_indices = cfg.get_subcarrier_indices()
    # 处理每个OFDM符号
    k = 0
    for i in range(cfg.num_symbols):
        # 检查当前符号是否需要插入导频
        if cfg.has_pilot(i):
            # 插入导频
            if cfg.display_est_result:
                print(f'insert pilot at {i} symbol')
            ofdm_symbol = insert_pilots(cfg)
        else:
            # 提取当前符号的比特
            start_idx = k * bits_per_symbol
            end_idx = start_idx + bits_per_symbol
            symbol_bits = bits[start_idx:end_idx]
            
            # QAM调制
            data_symbols = qam_modulation(symbol_bits, cfg.mod_order)
            # 不插入导频，所有子载波都用于数据传输
            
            ofdm_symbol = np.zeros(cfg.n_fft, dtype=np.complex64)
            ofdm_symbol[carrier_indices] = data_symbols
            k += 1
        # 保存频域符号
        freq_symbols[i] = ofdm_symbol
        
        # IFFT
        time_symbol = np.fft.ifft(ofdm_symbol, cfg.n_fft)
        
        # 添加循环前缀
        cp = time_symbol[-cfg.cp_len:]
        time_symbol = np.concatenate([cp, time_symbol])
        
        # 添加到总信号
        time_signal = np.concatenate([time_signal, time_symbol])
    
    return time_signal, freq_symbols

def plot_ofdm_symbol(ofdm_symbol: np.ndarray, pilot_indices: np.ndarray, 
                    data_indices: np.ndarray, title: str = "OFDM符号"):
    """绘制OFDM符号的星座图
    
    Args:
        ofdm_symbol: OFDM符号
        pilot_indices: 导频位置
        data_indices: 数据位置
        title: 图表标题
    """
    plt.figure(figsize=(10, 6))
    
    # 绘制导频符号
    plt.scatter(ofdm_symbol[pilot_indices].real, 
               ofdm_symbol[pilot_indices].imag,
               c='red', label='导频', marker='x')
    
    # 绘制数据符号
    plt.scatter(ofdm_symbol[data_indices].real,
               ofdm_symbol[data_indices].imag,
               c='blue', label='数据', marker='o')
    
    plt.grid(True)
    plt.axis('equal')
    plt.xlabel('实部')
    plt.ylabel('虚部')
    plt.title(title)
    plt.legend()
    plt.show()

def plot_ofdm_resource_grid(freq_symbols: np.ndarray, cfg: OFDMConfig, title: str = "OFDM资源网格"):
    """绘制OFDM资源网格
    
    Args:
        freq_symbols: 频域符号数组 [num_symbols, n_fft]
        pilot_indices: 导频位置
        data_indices: 数据位置
        cfg: 系统配置参数
        title: 图表标题
    """
    plt.figure(figsize=(12, 6))
    
    # 创建资源网格图
    grid = np.zeros((freq_symbols.shape[1], freq_symbols.shape[0]))
    pilot_indices = cfg.get_pilot_indices()
    data_indices = cfg.get_data_indices()
    # 标记导频和数据位置
    for i in range(freq_symbols.shape[0]):
        if cfg.has_pilot(i):
            grid[subcarrier_indices, i] = 1  # 导频
            # grid[data_indices, i] = 0.5  # 数据
        else:
            # 其他符号只包含数据
            subcarrier_indices = cfg.get_subcarrier_indices()
            grid[subcarrier_indices, i] = 0.5
    
    # 绘制资源网格
    plt.imshow(grid, aspect='auto', cmap='coolwarm')
    plt.colorbar(label='资源类型 (1:导频, 0.5:数据, 0:未使用)')
    
    # 设置坐标轴
    plt.xlabel('OFDM符号索引')
    plt.ylabel('子载波索引')
    plt.title(title)
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', label='导频'),
        Patch(facecolor='blue', label='未使用'),
        Patch(facecolor='white', label='数据')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    # 添加网格线
    plt.grid(True, color='gray', linestyle='--', alpha=0.5)
    
    plt.show()

if __name__ == "__main__":
    # 创建测试配置
    cfg = OFDMConfig(
        n_fft=4096,
        n_subcarrier=3276,
        cp_len=16,
        mod_order=4,  # 16QAM
        num_symbols=14,  # 14个OFDM符号
        pilot_pattern='comb',
        pilot_spacing=2,  # 频域间隔改为2
        pilot_symbols=[2,11]  # 只在第2和第11个符号上有导频
    )
    
    # 生成随机比特流
    np.random.seed(42)
    total_bits = cfg.get_total_bits()
    bits = np.random.randint(0, 2, size=total_bits)
    
    # 测试完整的OFDM发送处理
    print("\n测试OFDM发送处理...")
    time_signal, freq_symbols = ofdm_tx(bits, cfg)
    print(f"时域信号长度: {len(time_signal)}")
    print(f"时域信号功率: {np.mean(np.abs(time_signal)**2):.3f}")
    
    # 绘制时域信号
    plt.figure(figsize=(12, 4))
    plt.plot(np.abs(time_signal))
    plt.grid(True)
    plt.xlabel('采样点')
    plt.ylabel('幅度')
    plt.title('OFDM时域信号')
    plt.show()
    
    # 绘制资源网格
    plot_ofdm_resource_grid(freq_symbols, cfg,
                          "OFDM资源网格（红色为导频，蓝色为数据）")
    