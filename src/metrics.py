"""
性能指标计算模块
作者：AI助手
日期：2024-05-31
"""

import numpy as np
from typing import Tuple

def calculate_ber(bits_tx: np.ndarray, bits_rx: np.ndarray) -> float:
    """计算误比特率
    
    Args:
        bits_tx: 发送比特
        bits_rx: 接收比特
    
    Returns:
        误比特率
    """
    return np.sum(bits_tx != bits_rx) / len(bits_tx)

def calculate_ser(symbols_tx: np.ndarray, symbols_rx: np.ndarray) -> float:
    """计算误符号率
    
    Args:
        symbols_tx: 发送符号
        symbols_rx: 接收符号
    
    Returns:
        误符号率
    """
    return np.sum(symbols_tx != symbols_rx) / len(symbols_tx) 