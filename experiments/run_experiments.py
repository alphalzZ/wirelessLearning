"""
OFDM仿真实验运行脚本
作者：AI助手
日期：2024-05-31
"""

import numpy as np
import logging
from pathlib import Path
import sys
import pickle
# 添加项目根目录到Python路径
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.config import OFDMConfig, load_config
from src.ofdm_tx import ofdm_tx, add_timing_offset_and_freq_offset
from src.channel import awgn_channel, rayleigh_channel,multipath_channel, sionna_fading_channel, sionna_tdl_channel
from src.ofdm_rx import ofdm_rx
from src.metrics import calculate_ber, calculate_ser
from src.fec import compute_k
import matplotlib.pyplot as plt

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
    # 总bit
    k = compute_k(cfg, cfg.code_rate)
    # 生成随机比特流
    bits_tx = np.random.randint(0, 2, k)
    
    # 发送端处理
    tx_signal, _ = ofdm_tx(bits_tx, cfg)
    
    # 信道传输
    if cfg.channel_type == "awgn":
        rx_signal = awgn_channel(tx_signal, snr_db, num_rx=cfg.num_rx_ant)
    elif cfg.channel_type == "multipath":
        rx_signal, _ = multipath_channel(tx_signal, snr_db, num_rx=cfg.num_rx_ant)
    elif cfg.channel_type == "rayleigh":
        rx_signal, _ = rayleigh_channel(tx_signal, snr_db, num_rx=cfg.num_rx_ant)
    elif cfg.channel_type == "sionna_fading":
        rx_signal = sionna_fading_channel(tx_signal, snr_db, num_rx=cfg.num_rx_ant)
    elif cfg.channel_type == "sionna_tdl":
        rx_signal = sionna_tdl_channel(tx_signal, snr_db, num_rx=cfg.num_rx_ant)
    rx_signal = add_timing_offset_and_freq_offset(rx_signal, cfg)
    # 接收端处理
    _, bits_rx = ofdm_rx(rx_signal, cfg)
    
    # 计算性能指标
    ber = calculate_ber(bits_tx, bits_rx)

    return ber

def main():
    # 加载配置
    config_path = Path(__file__).parent.parent / "config.yaml"
    cfg = load_config(config_path)
    snr_db_list = np.arange(0, 31, 3)
    num_trials = 100  # 每个SNR点的仿真次数
    num_carriers = [1536, 3276]
    
    # 创建结果目录
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    # 创建结果存储字典
    all_results = {}
    
    # 为每种方法运行仿真
    for num_carrier in num_carriers:
        cfg.set_n_subcarrier(num_carrier)
        # 运行不同SNR下的实验
        results = []
        for snr_db in snr_db_list:
            # 存储当前SNR下的所有误码率
            current_ber = []
            
            # 蒙特卡洛仿真
            for trial in range(num_trials):
                ber = run_single_experiment(snr_db, cfg)
                current_ber.append(ber)
            # 计算平均误码率
            avg_ber = np.mean(current_ber)
            results.append((snr_db, avg_ber))
            logger.info(f"channel_type: {cfg.channel_type}, num_carrier: {num_carrier},"
                        f" SNR = {snr_db:2d} dB, BER = {avg_ber:.3e}")
            if avg_ber == 0:
                break
        # 存储结果
        all_results[num_carrier] = results
        
        # 保存结果到文件
        results_file = results_dir / f"results_{cfg.channel_type}_{num_carrier}.pkl"
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)
    
    # 绘制所有方法的BER vs SNR曲线
    plt.figure(figsize=(10, 5))
    for num_carrier, results in all_results.items():
        plt.semilogy(snr_db_list, [result[1] for result in results], 'o-', label=f'子载波{num_carrier}')
    
    plt.grid(True)
    plt.xlabel('SNR (dB)')
    plt.ylabel('BER')
    plt.title('不同子载波数的BER性能对比')
    plt.legend()
    
    # 保存图片
    plt.savefig(results_dir / f"ber_vs_snr_comparison_{cfg.channel_type}_{num_carrier}.png")
    plt.show()
    plt.close()

if __name__ == "__main__":
    main()