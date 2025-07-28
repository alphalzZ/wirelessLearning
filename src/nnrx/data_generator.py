"""
在线数据生成器模块
作者：AI助手
日期：2025-06-27
"""
import numpy as np
import tensorflow as tf
import sys
from typing import Tuple, Generator
from pathlib import Path

# 添加项目根目录到Python路径
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.config import OFDMConfig
from src.ofdm_tx import ofdm_tx, compute_k, add_timing_offset_and_freq_offset
from src.channel import awgn_channel, multipath_channel
from src.ofdm_rx import (
    remove_cp_and_fft,
    estimate_timing_offset,
    estimate_frequency_offset,
    compensate_frequency_offset,
    compensate_timing_offset,
    )

def create_data(cfg: OFDMConfig, batch_size: int) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    在线生成一批训练数据。

    Args:
        cfg (OFDMConfig): OFDM系统配置。
        batch_size (int): 每批数据的样本数。

    Yields:
        Generator[Tuple[np.ndarray, np.ndarray], None, None]: 
        一个生成器，每次产出一批训练数据。
        元组的第一个元素是接收到的时域信号 (X)，形状为 (batch_size, num_tx_ant, signal_len)。
        元组的第二个元素是原始的频域符号 (y)，形状为 (batch_size, num_tx_ant, num_symbols, n_fft)。
    """
    batch_count = 0  # 添加计数器
    while True:
        if batch_count % 100 == 0:  # 每100个批次打印一次随机种子
            print(f"生成新的批次 {batch_count}, 随机种子: {np.random.randint(0, 1000000)}")
        batch_count += 1
        
        batch_x = []
        batch_y = []
        for _ in range(batch_size):
            # 1. 生成随机比特
            k = compute_k(cfg, cfg.code_rate)
            bits = np.random.randint(0, 2, k)

            # 2. OFDM发送端处理，生成时域信号和频域符号
            tx_signal, freq_symbols = ofdm_tx(bits, cfg)

            # 3. 通过信道
            if cfg.channel_type == 'awgn':
                rx_signal = awgn_channel(tx_signal, num_rx=cfg.num_rx_ant, num_tx=cfg.num_tx_ant)
            elif cfg.channel_type == 'multipath':
                rx_signal, h = multipath_channel(tx_signal, num_rx=cfg.num_rx_ant, num_tx=cfg.num_tx_ant)
            else:
                # 可根据需要扩展其他信道模型
                raise ValueError(f"不支持的信道类型: {cfg.channel_type}")
            
            # 4. (可选) 添加频偏和时偏
            signal = add_timing_offset_and_freq_offset(rx_signal, cfg)

            if signal.ndim == 1:

                signal = signal[None, :]

            num_ant = signal.shape[0]

            # 1. 移除循环前缀并进行FFT
            rx_symbols = remove_cp_and_fft(signal, cfg)
            
            # 2. 使用导频进行频偏估计和补偿
            offset = cfg.get_subcarrier_offset()
            pilot_symbol_indices = cfg.get_pilot_symbol_indices()
            pilot_symbols = cfg.get_pilot_symbols(pilot_symbol_indices)#排列顺序为第一根天线第一个DMRS，第二根天线，第一个DMRS，第一根天线，第二个DMRS，...
            pilot_symbols = pilot_symbols.reshape(cfg.num_tx_ant,len(pilot_symbol_indices), -1)  # (num_tx_ant, num_pilots, n_pilots)
            pilot_indices = cfg.get_pilot_indices() - offset
            est_timing = []
            est_freq_offset = []
            for a in range(num_ant):
                est_timing.append(
                    estimate_timing_offset(rx_symbols[a], pilot_symbols, pilot_indices, cfg)
                )
                est_freq_offset.append(
                    estimate_frequency_offset(rx_symbols[a], pilot_symbols, pilot_indices, cfg)
                )
            if cfg.num_tx_ant == 1:
                est_time = np.array(est_timing)[:,None]
                est_freq = np.array(est_freq_offset)[:,None]
            else:
                est_time = np.array(est_timing)
                est_freq = np.array(est_freq_offset)
            est_timing = np.mean(est_time,axis=1)
            est_freq_offset = np.mean(est_freq, axis=1)
            if cfg.display_est_result:
                print(f"估计的时延: {est_timing}, 估计的频偏: {est_freq_offset}")

            # 逐层逐天线补偿频偏
            comp_ant = [
                compensate_frequency_offset(
                    signal[a],
                    est_freq_offset[a],
                    cfg,
                )
                for a in range(num_ant)
            ]
            signal_freq_comp = np.stack(comp_ant, axis=0)

            # CP 移除与FFT
            rx_symbols = remove_cp_and_fft(signal_freq_comp, cfg)

            # 逐层逐天线补偿时延
            comp_ant = [
                compensate_timing_offset(
                    rx_symbols[a],
                    est_timing[a],
                    cfg,
                )
                for a in range(num_ant)
            ]

            signal_timing = np.stack(comp_ant, axis=0) # (num_ant, num_symbols, n_subcarrier)
            real_part = np.real(signal_timing)
            imag_part = np.imag(signal_timing)
            batch_x.append(np.concatenate((real_part, imag_part), axis=0, dtype=np.float32))  # 将实部和虚部堆叠成最后一维
            batch_y.append(bits)  # 将比特转换为一行

        # 将列表转换为numpy数组
        X = np.array(batch_x,dtype=np.float32)  # (batch_size, 2*num_rx_ant, num_symbols, n_subcarrier)
        y = np.array(batch_y,dtype=np.float32)  # (batch_size, k)
        yield X, y

def create_tf_dataset(cfg: OFDMConfig, batch_size: int) -> tf.data.Dataset:
    """
    创建一个TensorFlow数据集。

    Args:
        cfg (OFDMConfig): OFDM系统配置。
        batch_size (int): 批大小。

    Returns:
        tf.data.Dataset: TensorFlow数据集。
    """
    # 获取接收信号和频域符号的形状和类型
    k = compute_k(cfg, cfg.code_rate)
    bits = np.random.randint(0, 2, k)
    tx_signal, freq_symbols = ofdm_tx(bits, cfg)
    
    output_signature = (
        tf.TensorSpec(shape=(batch_size, 2*cfg.num_rx_ant, cfg.num_symbols, cfg.n_subcarrier), dtype=np.float32),
        tf.TensorSpec(shape=(batch_size, k), dtype=np.float32)
    )

    dataset = tf.data.Dataset.from_generator(
        lambda: create_data(cfg, batch_size),
        output_signature=output_signature
    )
    
    return dataset

if __name__ == '__main__':
    # --- 使用示例 ---
    # 1. 创建一个OFDM配置
    config = OFDMConfig(
        n_fft=256,
        n_subcarrier=200,
        cp_len=16,
        mod_order=4,  # 16QAM
        num_symbols=14,
        num_tx_ant=1,
        num_rx_ant=2,
        snr_db=20,
        channel_type='multipath'
    )

    # 2. 创建TensorFlow数据集
    BATCH_SIZE = 16
    STEPS_PER_EPOCH = 100
    train_dataset = create_tf_dataset(config, BATCH_SIZE)

    # 3. 迭代数据集并检查数据形状
    for x_batch, y_batch in train_dataset.take(1):
        print("成功生成一批数据！")
        print(f"输入 (X) 形状: {x_batch.shape}")
        print(f"标签 (y) 形状: {y_batch.shape}")

    # 可以在这里将 `train_dataset` 传递给 `model.fit()`
    # model.fit(train_dataset, epochs=10)
    # model.fit(train_dataset, epochs=10)
