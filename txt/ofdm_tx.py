"""
OFDM���Ͷ˴���ģ��
���ߣ�AI����
���ڣ�2024-05-31
"""

import numpy as np
from typing import Tuple, Optional
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows ����
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # Windows ΢���ź�
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # macOS
# plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC']  # Linux

plt.rcParams['axes.unicode_minus'] = False  # ���������ʾ����

# �����Ŀ��Ŀ¼��Python·��
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.config import OFDMConfig

import numpy as np

def qam_modulation(bits: np.ndarray, Qm: int) -> np.ndarray:
    """
    5G?NR Gray?coded QAM (38.211?��5.1)
    Qm = 2 (QPSK) | 4 (16QAM) | 6 (64QAM)
    Returns power?normalized symbols (E{|d|^2}=1).
    """
    if Qm not in (2, 4, 6):
        raise ValueError("Qm must be 2, 4 or 6")
    if bits.size % Qm:
        raise ValueError(f"len(bits) must be a multiple of Qm={Qm}")

    b = bits.astype(np.int8).reshape(-1, Qm)   # ǿ�� 0/1 ����

    if Qm == 2:                     # QPSK
        i = 1 - 2 * b[:, 0]
        q = 1 - 2 * b[:, 1]
        norm = np.sqrt(2)

    elif Qm == 4:                   # 16?QAM  (��1, ��3)
        i = (1 - 2 * b[:, 0]) * (1 + 2 * b[:, 1])
        q = (1 - 2 * b[:, 2]) * (1 + 2 * b[:, 3])
        norm = np.sqrt(10)

    else:                           # 64?QAM  (��1, ��3, ��5, ��7)
        i = (1 - 2 * b[:, 0]) * (1 + 2 * b[:, 1] + 4 * b[:, 2])
        q = (1 - 2 * b[:, 3]) * (1 + 2 * b[:, 4] + 4 * b[:, 5])
        norm = np.sqrt(42)

    return (i + 1j * q) / norm


def insert_pilots(cfg: OFDMConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """���뵼Ƶ����
    
    Args:
        data_symbols: ���ݷ���
        cfg: ϵͳ���ò���
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (��Ƶ����, ��Ƶλ��, ����λ��)
    """
    # ��ȡ��Ƶλ�ú�����λ��
    pilot_indices = cfg.get_pilot_indices()
    
    # ���ɵ�Ƶ����
    pilot_symbols = cfg.get_pilot_symbols()
    
    # ����������OFDM����
    ofdm_symbol = np.zeros(cfg.n_fft, dtype=np.complex64)
    ofdm_symbol[pilot_indices] = pilot_symbols
    
    return ofdm_symbol

def ofdm_tx(bits: np.ndarray, cfg: OFDMConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """OFDM���Ͷ˴���
    
    Args:
        bits: ���������
        cfg: ϵͳ���ò���
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: (ʱ���ź�, ��Ƶλ��, ����λ��, Ƶ�����)
    """
    # ��֤����
    total_bits = cfg.get_total_bits()
    if len(bits) != total_bits:
        raise ValueError(f"������������ȱ�����{total_bits}")
    
    # ����ÿ��OFDM���ŵı�����
    bits_per_symbol = cfg.get_total_bits_per_symbol()
    
    # ��ʼ��ʱ���ź������Ƶ���������
    time_signal = np.array([], dtype=np.complex64)
    freq_symbols = np.zeros((cfg.num_symbols, cfg.n_fft), dtype=np.complex64)
    carrier_indices = cfg.get_subcarrier_indices()
    # ����ÿ��OFDM����
    k = 0
    for i in range(cfg.num_symbols):
        # ��鵱ǰ�����Ƿ���Ҫ���뵼Ƶ
        if cfg.has_pilot(i):
            # ���뵼Ƶ
            print(f'insert pilot at {i} symbol')
            ofdm_symbol = insert_pilots(cfg)
        else:
            # ��ȡ��ǰ���ŵı���
            start_idx = k * bits_per_symbol
            end_idx = start_idx + bits_per_symbol
            symbol_bits = bits[start_idx:end_idx]
            
            # QAM����
            data_symbols = qam_modulation(symbol_bits, cfg.mod_order)
            # �����뵼Ƶ���������ز����������ݴ���
            
            ofdm_symbol = np.zeros(cfg.n_fft, dtype=np.complex64)
            ofdm_symbol[carrier_indices] = data_symbols
            k += 1
        # ����Ƶ�����
        freq_symbols[i] = ofdm_symbol
        
        # IFFT
        time_symbol = np.fft.ifft(ofdm_symbol, cfg.n_fft)
        
        # ���ѭ��ǰ׺
        cp = time_symbol[-cfg.cp_len:]
        time_symbol = np.concatenate([cp, time_symbol])
        
        # ��ӵ����ź�
        time_signal = np.concatenate([time_signal, time_symbol])
    
    return time_signal, freq_symbols

def plot_ofdm_symbol(ofdm_symbol: np.ndarray, pilot_indices: np.ndarray, 
                    data_indices: np.ndarray, title: str = "OFDM����"):
    """����OFDM���ŵ�����ͼ
    
    Args:
        ofdm_symbol: OFDM����
        pilot_indices: ��Ƶλ��
        data_indices: ����λ��
        title: ͼ�����
    """
    plt.figure(figsize=(10, 6))
    
    # ���Ƶ�Ƶ����
    plt.scatter(ofdm_symbol[pilot_indices].real, 
               ofdm_symbol[pilot_indices].imag,
               c='red', label='��Ƶ', marker='x')
    
    # �������ݷ���
    plt.scatter(ofdm_symbol[data_indices].real,
               ofdm_symbol[data_indices].imag,
               c='blue', label='����', marker='o')
    
    plt.grid(True)
    plt.axis('equal')
    plt.xlabel('ʵ��')
    plt.ylabel('�鲿')
    plt.title(title)
    plt.legend()
    plt.show()

def plot_ofdm_resource_grid(freq_symbols: np.ndarray, cfg: OFDMConfig, title: str = "OFDM��Դ����"):
    """����OFDM��Դ����
    
    Args:
        freq_symbols: Ƶ��������� [num_symbols, n_fft]
        pilot_indices: ��Ƶλ��
        data_indices: ����λ��
        cfg: ϵͳ���ò���
        title: ͼ�����
    """
    plt.figure(figsize=(12, 6))
    
    # ������Դ����ͼ
    grid = np.zeros((freq_symbols.shape[1], freq_symbols.shape[0]))
    pilot_indices = cfg.get_pilot_indices()
    data_indices = cfg.get_data_indices()
    # ��ǵ�Ƶ������λ��
    for i in range(freq_symbols.shape[0]):
        if cfg.has_pilot(i):
            grid[subcarrier_indices, i] = 1  # ��Ƶ
            # grid[data_indices, i] = 0.5  # ����
        else:
            # ��������ֻ��������
            subcarrier_indices = cfg.get_subcarrier_indices()
            grid[subcarrier_indices, i] = 0.5
    
    # ������Դ����
    plt.imshow(grid, aspect='auto', cmap='coolwarm')
    plt.colorbar(label='��Դ���� (1:��Ƶ, 0.5:����, 0:δʹ��)')
    
    # ����������
    plt.xlabel('OFDM��������')
    plt.ylabel('���ز�����')
    plt.title(title)
    
    # ���ͼ��
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', label='��Ƶ'),
        Patch(facecolor='blue', label='δʹ��'),
        Patch(facecolor='white', label='����')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    # ���������
    plt.grid(True, color='gray', linestyle='--', alpha=0.5)
    
    plt.show()

if __name__ == "__main__":
    # ������������
    cfg = OFDMConfig(
        n_fft=4096,
        n_subcarrier=3276,
        cp_len=16,
        mod_order=4,  # 16QAM
        num_symbols=14,  # 14��OFDM����
        pilot_pattern='comb',
        pilot_spacing=2,  # Ƶ������Ϊ2
        pilot_symbols=[2,11]  # ֻ�ڵ�2�͵�11���������е�Ƶ
    )
    
    # �������������
    np.random.seed(42)
    total_bits = cfg.get_total_bits()
    bits = np.random.randint(0, 2, size=total_bits)
    
    # ����������OFDM���ʹ���
    print("\n����OFDM���ʹ���...")
    time_signal, freq_symbols = ofdm_tx(bits, cfg)
    print(f"ʱ���źų���: {len(time_signal)}")
    print(f"ʱ���źŹ���: {np.mean(np.abs(time_signal)**2):.3f}")
    
    # ����ʱ���ź�
    plt.figure(figsize=(12, 4))
    plt.plot(np.abs(time_signal))
    plt.grid(True)
    plt.xlabel('������')
    plt.ylabel('����')
    plt.title('OFDMʱ���ź�')
    plt.show()
    
    # ������Դ����
    plot_ofdm_resource_grid(freq_symbols, cfg,
                          "OFDM��Դ���񣨺�ɫΪ��Ƶ����ɫΪ���ݣ�")
    