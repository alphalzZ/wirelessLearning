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

from src.config import OFDMConfig,load_config
from src.ofdm_tx import qam_modulation, insert_pilots, ofdm_tx, add_timing_offset_and_freq_offset, compute_k
from src.ofdm_rx import (
    ofdm_rx,
    qam_demodulation,
    estimate_frequency_offset,
    compensate_frequency_offset,
    estimate_timing_offset,
    estimate_channel,
    channel_equalization,
    remove_cp_and_fft,
)
from src.channel import awgn_channel,rayleigh_channel, multipath_channel, sionna_fading_channel, sionna_tdl_channel

class TestOFDMSystem(unittest.TestCase):
    def setUp(self):
        """测试前的准备工作"""
        # 创建测试配置
        self.cfg = load_config('config.yaml')
        
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
        # 测试导频插入
        ofdm_symbol = insert_pilots(self.cfg)
        
        # 验证输出
        self.assertEqual(len(ofdm_symbol), self.cfg.n_fft)
        
    def test_ofdm_tx_rx(self):
        """测试OFDM收发端功能"""
        # 生成随机比特流
        k = compute_k(self.cfg, self.cfg.code_rate)
        bits = np.random.randint(0, 2, k)
        
        # 发送端处理
        tx_signal, freq_symbols = ofdm_tx(bits, self.cfg)
        
        # 添加噪声
        if self.cfg.channel_type == 'multipath':
            rx_signal, h_channel = multipath_channel(
                tx_signal, self.cfg.snr_db, num_rx=self.cfg.num_rx_ant
            )
        elif self.cfg.channel_type == 'awgn':
            rx_signal = awgn_channel(tx_signal, self.cfg.snr_db, num_rx=self.cfg.num_rx_ant)
        elif self.cfg.channel_type == 'rayleigh':
            rx_signal, h_channel = rayleigh_channel(
                tx_signal, self.cfg.snr_db, num_rx=self.cfg.num_rx_ant
            )
        elif self.cfg.channel_type == 'sionna_fading':
            rx_signal = sionna_fading_channel(
                tx_signal, self.cfg.snr_db, num_rx=self.cfg.num_rx_ant
            )
        elif self.cfg.channel_type == 'sionna_tdl':
            rx_signal = sionna_tdl_channel(
                tx_signal, self.cfg.snr_db, num_rx=self.cfg.num_rx_ant
            )
        else:
            raise ValueError(f"不支持的信道类型: {self.cfg.channel_type}")
        rx_signal = add_timing_offset_and_freq_offset(rx_signal, self.cfg)
        # 接收端处理
        rx_syms, rx_bits = ofdm_rx(rx_signal, self.cfg)
        # 计算误码率
        bits_error = np.mean(rx_bits != bits)
        if self.cfg.code_rate < 1.0:
            print(f'ldpc bit error:{bits_error}')
        else:
            print(f'hard decision bit error:{bits_error}')
        # 绘制星座图
        plt.figure(figsize=(10, 5))
        data_indices = self.cfg.get_data_symbol_indices()
        subcarrier_indices = self.cfg.get_subcarrier_indices()
        print(f'data loc is{data_indices}')
        # 绘制发送符号的星座图
        freq_symbols_plot = freq_symbols[np.ix_(data_indices,subcarrier_indices)]
        plt.subplot(121)
        plt.scatter(freq_symbols_plot.real, freq_symbols_plot.imag, c='b', marker='o', label='发送符号')
        plt.grid(True)
        plt.title('发送符号星座图')
        plt.xlabel('实部')
        plt.ylabel('虚部')
        plt.legend()
        
        # 绘制接收符号的星座图
        plt.subplot(122)
        plt.scatter(rx_syms[data_indices].real, rx_syms[data_indices].imag, c='r', marker='.', label='接收符号')
        plt.grid(True)
        plt.title('接收符号星座图')
        plt.xlabel('实部')
        plt.ylabel('虚部')
        plt.legend()

        plt.tight_layout()
        plt.show()
        plt.close()

    def test_multipath_channel(self):
        """测试多径信道功能"""
        # 生成测试信号
        tx = (np.random.randn(1000) + 1j * np.random.randn(1000)) / np.sqrt(2)
        snr_db_target = 20
        sigScale = 10**(snr_db_target/20)
        tx = tx * sigScale
        rx, h = multipath_channel(tx, snr_db_target, num_rx=self.cfg.num_rx_ant)
        meas_snr = np.mean(np.abs(np.convolve(tx, h,'same'))**2) / np.mean(np.abs(rx - np.convolve(tx,h,'same'))**2)
        print(f'multipath channel snr is {10*np.log10(meas_snr)} dB')   # 应接近 20 dB

    def test_multi_antenna_reception(self):
        """测试多天线接收流程"""
        self.cfg.num_rx_ant = 4
        k = compute_k(self.cfg, self.cfg.code_rate)
        bits = np.random.randint(0, 2, k)
        tx_signal, _ = ofdm_tx(bits, self.cfg)
        rx_signal = awgn_channel(tx_signal, self.cfg.snr_db, num_rx=self.cfg.num_rx_ant)
        _, rx_bits = ofdm_rx(rx_signal, self.cfg)
        self.assertEqual(len(rx_bits), len(bits))

    def test_ofdm_tx_rx_ldpc(self):
        """结合LDPC编译码的完整OFDM流程"""
        rate = self.cfg.code_rate
        k = compute_k(self.cfg, rate)
        info_bits = np.random.randint(0, 2, k)

        tx_signal, _ = ofdm_tx(info_bits, self.cfg)
        rx_signal = awgn_channel(tx_signal, self.cfg.snr_db, num_rx=self.cfg.num_rx_ant)
        _, dec_bits = ofdm_rx(rx_signal, self.cfg)

        ber = np.mean(dec_bits != info_bits)
        print('LDPC BER:', ber)
        self.assertEqual(len(dec_bits), len(info_bits))


    def test_ofdm_rx_with_channel_estimation(self):
        """测试OFDM接收端信道估计功能"""
        # 生成随机比特流
        np.random.seed(42)
        k = compute_k(self.cfg, self.cfg.code_rate)
        bits = np.random.randint(0, 2, k)
        
        # 生成OFDM符号
        time_signal, freq_symbols = ofdm_tx(bits, self.cfg)
        
        # 添加AWGN噪声
        snr_db = self.cfg.snr_db
        #计算噪声线性值
        noise_var = 10 ** (-snr_db / 10)
        rx_signal = awgn_channel(
            time_signal, snr_db, num_rx=self.cfg.num_rx_ant
        )

        # 移除CP并进行FFT
        rx_symbols_freq_offset = remove_cp_and_fft(rx_signal, self.cfg)

        # 估计信道
        h_est = estimate_channel(rx_symbols_freq_offset, self.cfg)
        #信道均衡
        rx_symbols_equalized = channel_equalization(rx_symbols_freq_offset, h_est, noise_var)
        # 画出均衡后的信号和实际信号的星座图
        plt.figure()
        plt.subplot(131)
        plt.scatter(rx_symbols_freq_offset.real, rx_symbols_freq_offset.imag,label='接收信号')
        plt.subplot(132)
        plt.scatter(rx_symbols_equalized.real, rx_symbols_equalized.imag,label='均衡后')
        plt.subplot(133)
        plt.scatter(freq_symbols.real, freq_symbols.imag,label='实际')
        plt.legend()
        plt.show()
        plt.close()
        print(f"信道估计结果: {h_est}")


if __name__ == '__main__':
    # 创建测试套件
    # unittest.main()
    suite = unittest.TestSuite()
    # 只添加信道估计测试
    suite.addTest(TestOFDMSystem('test_ofdm_tx_rx'))
    # 运行测试
    runner = unittest.TextTestRunner()
    runner.run(suite)
