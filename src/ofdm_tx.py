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

def qam_modulation(bits: np.ndarray, mod_order: int) -> np.ndarray:
    """QAM调制
    
    Args:
        bits: 输入比特流
        mod_order: 调制阶数（2:QPSK, 4:16QAM, 6:64QAM）
        
    Returns:
        调制后的符号
    """
    # 验证输入
    if mod_order not in [2, 4, 6]:
        raise ValueError("调制阶数必须是2、4或6")
    if len(bits) % mod_order != 0:
        raise ValueError(f"比特流长度必须是{mod_order}的倍数")
    
    # 将比特流重塑为mod_order比特一组
    bits_reshaped = bits.reshape(-1, mod_order)
    
    # 根据调制阶数选择映射方式
    if mod_order == 2:  # QPSK
        symbols = (1 - 2 * bits_reshaped[:, 0]) + 1j * (1 - 2 * bits_reshaped[:, 1])
        norm_factor = np.sqrt(2)
    elif mod_order == 4:  # 16QAM
        real = (1 - 2 * bits_reshaped[:, 0]) * (2 - bits_reshaped[:, 1])
        imag = (1 - 2 * bits_reshaped[:, 2]) * (2 - bits_reshaped[:, 3])
        symbols = real + 1j * imag
        norm_factor = np.sqrt(10)
    else:  # 64QAM
        real = (1 - 2 * bits_reshaped[:, 0]) * (4 - 2 * bits_reshaped[:, 1] - bits_reshaped[:, 2])
        imag = (1 - 2 * bits_reshaped[:, 3]) * (4 - 2 * bits_reshaped[:, 4] - bits_reshaped[:, 5])
        symbols = real + 1j * imag
        norm_factor = np.sqrt(42)
    
    # 功率归一化
    return symbols / norm_factor

def insert_pilots(data_symbols: np.ndarray, cfg: OFDMConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """插入导频符号
    
    Args:
        data_symbols: 数据符号
        cfg: 系统配置参数
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (导频符号, 导频位置, 数据位置)
    """
    # 获取导频位置和数据位置
    pilot_indices = cfg.get_pilot_indices()
    data_indices = cfg.get_data_indices()
    
    # 验证数据符号数量
    if len(data_symbols) != len(data_indices):
        raise ValueError(f"数据符号数量({len(data_symbols)})与数据子载波数量({len(data_indices)})不匹配")
    
    # 生成导频符号
    pilot_symbols = cfg.get_pilot_symbols()
    
    # 创建完整的OFDM符号
    ofdm_symbol = np.zeros(cfg.n_fft, dtype=np.complex64)
    ofdm_symbol[pilot_indices] = pilot_symbols
    ofdm_symbol[data_indices] = data_symbols
    
    return ofdm_symbol, pilot_indices, data_indices

def ofdm_tx(bits: np.ndarray, cfg: OFDMConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """OFDM发送端处理
    
    Args:
        bits: 输入比特流
        cfg: 系统配置参数
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: (时域信号, 导频位置, 数据位置, 频域符号)
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
    pilot_indices = None
    data_indices = None
    
    # 处理每个OFDM符号
    for i in range(cfg.num_symbols):
        # 提取当前符号的比特
        start_idx = i * bits_per_symbol
        end_idx = start_idx + bits_per_symbol
        symbol_bits = bits[start_idx:end_idx]
        
        # QAM调制
        data_symbols = qam_modulation(symbol_bits, cfg.mod_order)
        
        # 检查当前符号是否需要插入导频
        if cfg.has_pilot(i):
            # 插入导频
            ofdm_symbol, pilot_indices, data_indices = insert_pilots(data_symbols, cfg)
        else:
            # 不插入导频，所有子载波都用于数据传输
            ofdm_symbol = np.zeros(cfg.n_fft, dtype=np.complex64)
            ofdm_symbol[:len(data_symbols)] = data_symbols
        
        # 保存频域符号
        freq_symbols[i] = ofdm_symbol
        
        # IFFT
        time_symbol = np.fft.ifft(ofdm_symbol, cfg.n_fft)
        
        # 添加循环前缀
        cp = time_symbol[-cfg.cp_len:]
        time_symbol = np.concatenate([cp, time_symbol])
        
        # 添加到总信号
        time_signal = np.concatenate([time_signal, time_symbol])
    
    return time_signal, pilot_indices, data_indices, freq_symbols

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

def plot_ofdm_resource_grid(freq_symbols: np.ndarray, pilot_indices: np.ndarray, 
                          data_indices: np.ndarray, cfg: OFDMConfig, title: str = "OFDM资源网格"):
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
    
    # 标记导频和数据位置
    for i in range(freq_symbols.shape[0]):
        if cfg.has_pilot(i):
            grid[pilot_indices, i] = 1  # 导频
            grid[data_indices, i] = 0.5  # 数据
        else:
            # 其他符号只包含数据
            all_data_indices = np.arange(freq_symbols.shape[1])
            grid[all_data_indices, i] = 0.5
    
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
        Patch(facecolor='blue', label='数据'),
        Patch(facecolor='white', label='未使用')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    # 添加网格线
    plt.grid(True, color='gray', linestyle='--', alpha=0.5)
    
    plt.show()

if __name__ == "__main__":
    # 创建测试配置
    cfg = OFDMConfig(
        n_fft=128,
        cp_len=16,
        mod_order=4,  # 16QAM
        num_symbols=14,  # 14个OFDM符号
        pilot_pattern='comb',
        pilot_spacing=2,  # 频域间隔改为2
        pilot_symbols=[2,11]  # 只在第2和第11个符号上有导频
    )
    
    # 生成随机比特流
    np.random.seed(42)
    bits = np.random.randint(0, 2, size=cfg.get_total_bits())
    
    # 测试QAM调制
    print("测试QAM调制...")
    data_symbols = qam_modulation(bits, cfg.mod_order)
    print(f"数据符号形状: {data_symbols.shape}")
    print(f"数据符号示例: {data_symbols[:5]}")
    print(f"数据符号功率: {np.mean(np.abs(data_symbols)**2):.3f}")
    
    # 测试导频插入
    print("\n测试导频插入...")
    # 只使用第一个OFDM符号的数据进行测试
    data_indices = cfg.get_data_indices()
    test_data_symbols = data_symbols[:len(data_indices)]
    ofdm_symbol, pilot_indices, data_indices = insert_pilots(test_data_symbols, cfg)
    print(f"导频位置: {pilot_indices}")
    print(f"导频数量: {len(pilot_indices)}")
    print(f"数据位置: {data_indices}")
    print(f"数据数量: {len(data_indices)}")
    
    # 可视化OFDM符号
    plot_ofdm_symbol(ofdm_symbol, pilot_indices, data_indices, 
                    "OFDM符号星座图（红色x为导频，蓝色o为数据）")
    
    # 测试完整的OFDM发送处理
    print("\n测试OFDM发送处理...")
    time_signal, pilot_indices, data_indices, freq_symbols = ofdm_tx(bits, cfg)
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
    plot_ofdm_resource_grid(freq_symbols, pilot_indices, data_indices, cfg,
                          "OFDM资源网格（红色为导频，蓝色为数据）")
    