"""
OFDM系统测试模块
作者：AI助手
日期：2024-05-31
"""

import unittest
import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# 添加项目根目录到Python路径
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.config import OFDMConfig
from src.ofdm_tx import qam_modulation, insert_pilots, ofdm_tx
from src.ofdm_rx import ofdm_rx, estimate_frequency_offset, compensate_frequency_offset, estimate_timing_offset, estimate_channel, channel_equalization, remove_cp_and_fft
from src.channel import awgn_channel,rayleigh_channel, multipath_channel

class TestOFDMSystem(unittest.TestCase):
    def setUp(self):
        """测试前的准备工作"""
        # 创建测试配置
        self.cfg = OFDMConfig(
            n_fft=256,
            cp_len=36,
            mod_order=4,  # 16QAM
            num_symbols=14,  # 测试用较少的符号数
            pilot_pattern='comb',
            pilot_spacing=2,  # 导频间隔
            pilot_symbols=[2,11],  # 在第2和第11个符号上插入导频
            timing_offset=10,
            freq_offset= 0.02
        )
        
    def test_qam_modulation(self):
        """测试QAM调制功能"""
        # 测试QPSK调制
        bits = np.array([0, 1, 1, 0])
        symbols = qam_modulation(bits, 2)
        self.assertEqual(len(symbols), 2)
        self.assertTrue(np.all(np.abs(symbols) > 0))
        
        # 测试16QAM调制
        bits = np.array([0, 1, 1, 0, 1, 0, 0, 1])
        symbols = qam_modulation(bits, 4)
        self.assertEqual(len(symbols), 2)
        self.assertTrue(np.all(np.abs(symbols) > 0))
        
        # 测试无效输入
        with self.assertRaises(ValueError):
            qam_modulation(bits, 3)  # 无效的调制阶数
            
    def test_insert_pilots(self):
        """测试导频插入功能"""
        # 生成测试数据
        data_symbols = np.ones(self.cfg.get_data_indices().shape[0], dtype=np.complex64)
        
        # 测试导频插入
        ofdm_symbol, pilot_indices, data_indices = insert_pilots(data_symbols, self.cfg)
        
        # 验证输出
        self.assertEqual(len(ofdm_symbol), self.cfg.n_fft)
        self.assertTrue(np.all(np.abs(ofdm_symbol[pilot_indices]) > 0))
        self.assertTrue(np.all(np.abs(ofdm_symbol[data_indices]) > 0))
        
    def test_ofdm_tx_rx(self):
        """测试OFDM收发端功能"""
        # 生成随机比特流
        total_bits = self.cfg.get_total_bits()
        bits = np.random.randint(0, 2, total_bits)
        
        # 发送端处理
        tx_signal, pilot_indices, data_indices, freq_symbols = ofdm_tx(bits, self.cfg)
        
        # 添加噪声
        snr_db = 35
        #计算噪声线性值
        noise_var = 10 ** (-snr_db / 20)
        rx_signal,h_channel = multipath_channel(tx_signal, snr_db)   
        # rx_signal = awgn_channel(tx_signal,snr_db)
        phase_rotation = 2 * np.pi * self.cfg.freq_offset * np.arange(len(rx_signal)) / self.cfg.n_fft
        rx_signal = rx_signal * np.exp(1j * phase_rotation)
        rx_signal = np.roll(rx_signal, self.cfg.timing_offset)
        # 接收端处理
        self.cfg.noise_var = noise_var
        rx_syms = ofdm_rx(rx_signal, self.cfg)
        
        # 绘制星座图
        plt.figure(figsize=(10, 5))
        
        # 绘制发送符号的星座图
        plt.subplot(121)
        plt.scatter(freq_symbols.real, freq_symbols.imag, c='b', marker='o', label='发送符号')
        plt.grid(True)
        plt.title('发送符号星座图')
        plt.xlabel('实部')
        plt.ylabel('虚部')
        plt.legend()
        
        # 绘制接收符号的星座图
        plt.subplot(122)
        plt.scatter(rx_syms.real, rx_syms.imag, c='r', marker='x', label='接收符号')
        plt.grid(True)
        plt.title('接收符号星座图')
        plt.xlabel('实部')
        plt.ylabel('虚部')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        

    def test_multipath_channel(self):
        """测试多径信道功能"""
        # 生成测试信号
        signal = np.random.randn(1000) + 1j * np.random.randn(1000)
        
        # 测试多径信道
        rx_signal = multipath_channel(signal, self.cfg)
        
        # 验证输出
        self.assertEqual(len(rx_signal), len(signal))
        self.assertTrue(np.all(np.abs(rx_signal) > 0))

    def test_ofdm_rx_with_channel_estimation(self):
        """测试OFDM接收端信道估计功能"""
        # 生成随机比特流
        np.random.seed(42)
        total_bits = self.cfg.get_total_bits()
        bits = np.random.randint(0, 2, total_bits)
        
        # 生成OFDM符号
        time_signal, pilot_indices, data_indices, freq_symbols = ofdm_tx(bits, self.cfg)
        
        # 添加AWGN噪声
        snr_db = 0
        #计算噪声线性值
        noise_var = 10 ** (-snr_db / 10)
        rx_signal,h_channel = multipath_channel(time_signal, snr_db)   

        # 移除CP并进行FFT
        rx_symbols_freq_offset = remove_cp_and_fft(rx_signal, self.cfg)

        # 估计信道
        h_est = estimate_channel(rx_symbols_freq_offset, self.cfg)
        #信道均衡
        rx_symbols_equalized = channel_equalization(rx_symbols_freq_offset, h_est, noise_var)
        # 画出均衡后的信号和实际信号的星座图
        plt.figure()
        plt.scatter(rx_symbols_freq_offset.real, rx_symbols_freq_offset.imag,label='接收信号')
        plt.scatter(rx_symbols_equalized.real, rx_symbols_equalized.imag,label='均衡后')
        plt.scatter(freq_symbols.real, freq_symbols.imag,label='实际')
        plt.legend()
        plt.show()
        print(f"信道估计误差: {h_est}")


if __name__ == '__main__':
    # 创建测试套件
    suite = unittest.TestSuite()
    # 只添加信道估计测试
    suite.addTest(TestOFDMSystem('test_ofdm_tx_rx'))
    # 运行测试
    runner = unittest.TextTestRunner()
    runner.run(suite)