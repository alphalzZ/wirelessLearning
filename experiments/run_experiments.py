"""
OFDM仿真实验运行脚本
作者：AI助手
日期：2024-05-31
"""

import numpy as np
import logging
from pathlib import Path
from src.config import OFDMConfig, load_config
from src.ofdm_tx import ofdm_tx
from src.channel import awgn_channel, rayleigh_channel
from src.ofdm_rx import ofdm_rx
from src.metrics import calculate_ber, calculate_ser

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_single_experiment(snr_db: float, cfg: OFDMConfig) -> float:
    """运行单次实验
    
    Args:
        snr_db: 信噪比(dB)
        cfg: 系统配置参数
    
    Returns:
        误比特率
    """
    # 生成随机比特流
    bits_tx = np.random.randint(0, 2, cfg.num_symbols * cfg.n_fft * cfg.mod_order)
    
    # 发送端处理
    tx_signal = ofdm_tx(bits_tx, cfg)
    
    # 信道传输
    if cfg.channel_type == "awgn":
        rx_signal = awgn_channel(tx_signal, snr_db)
    else:
        rx_signal, _ = rayleigh_channel(tx_signal, snr_db)
    
    # 接收端处理
    bits_rx = ofdm_rx(rx_signal, cfg)
    
    # 计算性能指标
    ber = calculate_ber(bits_tx, bits_rx)
    return ber

def main():
    # 加载配置
    config_path = Path(__file__).parent.parent / "config.yaml"
    cfg = load_config(config_path)
    
    # 运行不同SNR下的实验
    results = []
    for snr_db in cfg.snr_db_list:
        ber = run_single_experiment(snr_db, cfg)
        results.append((snr_db, ber))
        logger.info(f"SNR = {snr_db:2d} dB, BER = {ber:.3e}")
    
    # TODO: 绘制BER曲线

if __name__ == "__main__":
    main() 