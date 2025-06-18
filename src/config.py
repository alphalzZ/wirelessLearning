"""
OFDM系统配置模块
作者：AI助手
日期：2024-05-31
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import numpy as np
import yaml

@dataclass
class OFDMConfig:
    """OFDM系统配置参数"""
    # Tx配置
    snr_db: float = 0                  # 信噪比
    freq_offset: float = 0.0           # 初始频偏估计值
    timing_offset: int = 0             # 初始定时偏移估计值
    est_time: str = 'fft_ml'           # 定时偏移估计方法：'fft_ml'（FFT最大似然）/'diff_phase'（相位差）或'ml_then_phase'（两步法）
    channel_type: str = 'awgn'         # 信道类型：'awgn'（高斯白噪声）或'multipath'（多径衰落）
    display_est_result: bool = False    # 是否显示信道估计结果
    # 基本参数
    n_fft: int = 4096                    # FFT大小
    cp_len: int = 320                   # 循环前缀长度
    n_subcarrier: int = 3276           # 子载波数量
    mod_order: int = 4                 # 调制阶数（2:QPSK, 4:16QAM, 6:64QAM）
    num_symbols: int = 14             # OFDM符号数量
    num_rx_ant: int = 1               # 接收天线数量
    code_rate: float = 0.5             # 信道编码码率 (0<rate<=1)
    
    # 导频配置
    pilot_pattern: str = 'comb'        # 导频图案类型：'comb'（梳状）或'block'（块状）
    pilot_spacing: int = 2             # 导频间隔（梳状导频）
    pilot_blocks: int = 4              # 导频块数量（块状导频）
    pilot_power: float = 1.0           # 导频功率（相对于数据符号）
    pilot_symbols: Optional[List[int]] = field(default_factory=lambda: [2,11])    # 包含导频的OFDM符号索引列表，None表示所有符号都包含导频
    pilot_period: int = 1              # 导频周期（每隔多少个OFDM符号插入一次导频）
    
    # 信道估计配置
    est_method: str = 'linear'         # 信道估计方法：'linear'（线性插值）或'ls'（最小二乘）
    interp_method: str = 'linear'      # 信道插值方式：'linear'或'nearest'
    est_time: str = 'fft_ml'           # 定时偏移估计方法：'fft_ml'（FFT最大似然）/'diff_phase'（相位差）或'ml_then_phase'（两步法）
    equ_method: str = 'mmse'          # 信道均衡方法：'mmse'（最小均方误差）或'mrc'（最大比率合并）或'irc'
    # 同步配置
    sync_method: str = 'auto'          # 同步方法：'auto'（自动）或'manual'（手动）

    _pilot_symbols_cache: Optional[dict[int, np.ndarray]] = None  # 按符号缓存导频
    def __post_init__(self):
        """初始化后处理"""
        # 验证基本参数
        if self.n_fft <= 0 or not self._is_power_of_2(self.n_fft):
            raise ValueError("FFT大小必须是2的幂")
        if self.n_fft < self.n_subcarrier+2*self.cp_len:
            raise ValueError("FFT大小必须大于子载波数量+2*循环前缀长度")
        if self.cp_len <= 0:
            raise ValueError("循环前缀长度必须大于0")
        if self.mod_order not in [2, 4, 6]:
            raise ValueError("调制阶数必须是2、4或6")
        if self.num_symbols <= 0:
            raise ValueError("OFDM符号数量必须大于0")
        if self.num_rx_ant <= 0:
            raise ValueError("接收天线数量必须大于0")
        if not (0 < self.code_rate <= 1):
            raise ValueError("code_rate 必须在0到1之间")
            
        # 验证导频配置
        if self.pilot_pattern not in ['comb', 'block']:
            raise ValueError("导频图案类型必须是'comb'或'block'")
        if self.pilot_spacing <= 0:
            raise ValueError("导频间隔必须大于0")
        if self.pilot_blocks <= 0:
            raise ValueError("导频块数量必须大于0")
        if self.pilot_power <= 0:
            raise ValueError("导频功率必须大于0")
        if self.pilot_period <= 0:
            raise ValueError("导频周期必须大于0")
        if self.pilot_symbols is not None:
            if not all(0 <= idx < self.num_symbols for idx in self.pilot_symbols):
                raise ValueError("导频符号索引必须在有效范围内")
            if len(set(self.pilot_symbols)) != len(self.pilot_symbols):
                raise ValueError("导频符号索引不能重复")
            
        # 验证信道估计配置
        if self.est_method not in ['linear', 'ls']:
            raise ValueError("信道估计方法必须是'linear'或'ls'")
        if self.interp_method not in ['linear', 'nearest']:
            raise ValueError("插值方式必须是'linear'或'nearest'")
            
        # 验证同步配置
        if self.sync_method not in ['auto', 'manual']:
            raise ValueError("同步方法必须是'auto'或'manual'")
    
    def _is_power_of_2(self, n: int) -> bool:
        """检查一个数是否是2的幂"""
        return n > 0 and (n & (n - 1)) == 0
    
    def set_n_subcarrier(self, n_subcarrier: int):
        self.n_subcarrier = n_subcarrier
        self._pilot_symbols_cache = None # 修改子载波数后，导频符号缓存失效
        self.get_pilot_symbols() # 重新生成导频符号

    def get_subcarrier_offset(self)->int:
        """获取子载波偏移"""
        return (self.n_fft-self.n_subcarrier)//2
    
    def get_subcarrier_indices(self)->np.ndarray:
        """获取子载波的数据位置"""
        offset = self.get_subcarrier_offset()
        return np.arange(offset, offset+self.n_subcarrier)

    def get_pilot_symbol_indices(self) -> np.ndarray:
        """获取包含导频的OFDM符号索引
        
        Returns:
            包含导频的OFDM符号索引数组
        """
        if self.pilot_symbols is not None:
            return np.array(self.pilot_symbols)
        else:
            return np.arange(0, self.num_symbols, self.pilot_period)
    
    def has_pilot(self, symbol_idx: int) -> bool:
        """检查指定的OFDM符号是否包含导频
        
        Args:
            symbol_idx: OFDM符号索引
            
        Returns:
            是否包含导频
        """
        if self.pilot_symbols is not None:
            return symbol_idx in self.pilot_symbols
        else:
            return symbol_idx % self.pilot_period == 0
    
    def get_pilot_indices(self) -> np.ndarray:
        """获取导频位置索引
        
        Returns:
            导频位置索引数组
        """
        if self.pilot_pattern == 'comb':
            # 梳状导频
            offset = self.get_subcarrier_offset()
            return np.arange(offset, self.n_fft-offset, self.pilot_spacing)
        else:
            # 块状导频
            block_size = self.n_fft // self.pilot_blocks
            indices = []
            for i in range(self.pilot_blocks):
                start = i * block_size
                indices.extend(range(start, start + block_size // 4))
            return np.array(indices)
    
    def get_pilot_symbols(self, symbol_idx: int | np.ndarray | None = None) -> np.ndarray:
        """获取指定 OFDM 符号的导频序列

        若 ``symbol_idx`` 为 ``None``，返回所有导频符号对应的序列矩阵。
        ``symbol_idx`` 可以是单个整数或整数数组。
        """
        if self._pilot_symbols_cache is None:
            self._pilot_symbols_cache = {}

        def gen(idx: int) -> np.ndarray:
            rng = np.random.default_rng(idx)
            pilot_indices = self.get_pilot_indices()
            bits = rng.integers(0, 2, size=len(pilot_indices) * 2, dtype=np.int8)
            syms = self._qpsk_modulate(bits) * np.sqrt(self.pilot_power)
            return syms

        if symbol_idx is None:
            idx_list = self.get_pilot_symbol_indices()
        else:
            idx_list = np.atleast_1d(symbol_idx).astype(int)

        out = []
        for idx in idx_list:
            if idx not in self._pilot_symbols_cache:
                self._pilot_symbols_cache[idx] = gen(int(idx))
            out.append(self._pilot_symbols_cache[idx])

        if np.isscalar(symbol_idx):
            return out[0]
        else:
            return np.stack(out, axis=0)
    
    def _qpsk_modulate(self, bits: np.ndarray) -> np.ndarray:
        """QPSK调制
        
        Args:
            bits: 输入比特流
            
        Returns:
            调制后的符号
        """
        # 将比特流重塑为2比特一组
        bits_reshaped = bits.reshape(-1, 2)
        # QPSK映射
        symbols = (1 - 2 * bits_reshaped[:, 0]) + 1j * (1 - 2 * bits_reshaped[:, 1])
        # 功率归一化
        return symbols / np.sqrt(2)
    
    def get_data_indices(self) -> np.ndarray:
        """获取数据子载波位置索引
        
        Returns:
            数据子载波位置索引数组
        """
        offset = self.get_subcarrier_offset()
        pilot_indices = self.get_pilot_indices()
        all_indices = np.arange(offset, offset+self.n_subcarrier)
        return np.setdiff1d(all_indices, pilot_indices)
    
    def get_data_symbol_indices(self) -> np.ndarray:
        """获取数据子载波位置索引
        
        Returns:
            数据子载波位置索引数组
        """
        pilot_indices = self.get_pilot_symbol_indices()
        all_indices = np.arange(self.num_symbols)
        return np.setdiff1d(all_indices, pilot_indices)
    
    def get_num_data_carriers(self) -> int:
        """获取数据子载波数量
        
        Returns:
            数据子载波数量
        """
        return self.n_subcarrier
    
    def get_num_pilot_carriers(self) -> int:
        """获取导频子载波数量
        
        Returns:
            导频子载波数量
        """
        return len(self.get_pilot_indices())
    
    def get_total_bits_per_symbol(self) -> int:
        """获取每个OFDM符号的总比特数
        
        Returns:
            总比特数
        """
        return self.get_num_data_carriers() * self.mod_order
    
    def get_total_bits(self) -> int:
        """获取整个传输的总比特数
        
        Returns:
            总比特数
        """
        return self.get_total_bits_per_symbol()*len(self.get_data_symbol_indices())

def load_config(config_path: str) -> OFDMConfig:
    """从YAML文件加载配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    return OFDMConfig(**config_dict) 

if __name__ == '__main__':
    cfg = load_config(r'.\config.yaml')
    print(cfg.num_symbols)