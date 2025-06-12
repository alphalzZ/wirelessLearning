"""
Forward Error Correction (FEC) utilities using Sionna's LDPC implementation.

该模块提供基于 Sionna 0.19.2 的 LDPC 信道编码和译码封装，支持根
据码率自动计算信息比特数，并在需要时进行分段处理。
"""

from __future__ import annotations

import math
from typing import List

import numpy as np

try:
    from sionna.fec.ldpc.encoding import LDPC5GEncoder
    from sionna.fec.ldpc.decoding import LDPC5GDecoder
except ImportError as exc:  # pragma: no cover - library may be missing
    raise ImportError(
        "Sionna 0.19.2 is required for LDPC operations"
    ) from exc

from .config import OFDMConfig

# 3GPP NR LDPC 最大信息比特数
MAX_LDPC_K = 8448
# 3GPP NR LDPC 最大码字长度 (来自 Sionna 限制)
MAX_LDPC_N = 316 * 384


def compute_k(cfg: OFDMConfig, rate: float) -> int:
    """根据码率计算信息比特数"""
    total_bits = cfg.get_total_bits()
    return int(total_bits * rate)


def get_segment_lengths(cfg: OFDMConfig, rate: float) -> tuple[list[int], list[int]]:
    """Return per-segment (k, n) lengths for a given configuration."""
    k_total = compute_k(cfg, rate)
    n_total = cfg.get_total_bits()

    if k_total <= MAX_LDPC_K and n_total <= MAX_LDPC_N:
        return [k_total], [n_total]

    num_segments = math.ceil(k_total / MAX_LDPC_K)
    base_k = k_total // num_segments
    k_segments = [base_k + (1 if i < k_total % num_segments else 0)
                  for i in range(num_segments)]

    base_n = n_total // num_segments
    n_segments = [base_n + (1 if i < n_total % num_segments else 0)
                  for i in range(num_segments)]

    return k_segments, n_segments


def _segment_information(bits: np.ndarray, segment_length: int) -> List[np.ndarray]:
    """将信息比特按照给定长度分段"""
    segments = []
    start = 0
    while start < len(bits):
        end = min(start + segment_length, bits.size)
        segments.append(bits[start:end])
        start = end
    return segments


def ldpc_encode(bits: np.ndarray, cfg: OFDMConfig, rate: float) -> List[np.ndarray]:
    """按需分段的 LDPC 编码

    参数``bits``长度应等于 ``compute_k(cfg, rate)``，编码后总长度与
    ``cfg.get_total_bits()`` 一致。若信息比特数超过 ``MAX_LDPC_K``，
    会自动将其平均分段。
    """

    k_total = compute_k(cfg, rate)
    n_total = cfg.get_total_bits()
    if bits.size != k_total:
        raise ValueError(f"输入比特数应为{k_total}, 实际为{bits.size}")

    k_segments, n_segments = get_segment_lengths(cfg, rate)

    encoded_segments = []
    start = 0
    for k_seg, n_seg in zip(k_segments, n_segments):
        seg_bits = bits[start:start + k_seg]
        start += k_seg
        encoder = LDPC5GEncoder(k_seg, n_seg)
        coded = encoder(seg_bits[None, :].astype("float32")).numpy().squeeze()
        encoded_segments.append(coded)

    return encoded_segments


def ldpc_decode(
    llrs: List[np.ndarray],
    cfg: OFDMConfig,
    rate: float,
    *,
    num_iter: int = 6,
) -> np.ndarray:
    """对应 :func:`ldpc_encode` 的分段译码"""

    k_total = compute_k(cfg, rate)
    k_segments, n_segments = get_segment_lengths(cfg, rate)

    if len(llrs) != len(n_segments):
        raise ValueError("LLR 段数与编码段数不匹配")

    decoded_list = []
    for llr, k_seg, n_seg in zip(llrs, k_segments, n_segments):
        if llr.size != n_seg:
            raise ValueError("LLR 长度与码字长度不符")
        encoder = LDPC5GEncoder(k_seg, n_seg)
        decoder = LDPC5GDecoder(encoder, hard_out=True, num_iter=num_iter)
        decoded = decoder(llr[None, :].astype("float32")).numpy().squeeze()
        decoded_list.append(decoded.astype(np.int8))

    decoded_bits = np.concatenate(decoded_list)[:k_total]
    return decoded_bits


if __name__ == "__main__":  # 简单测试
    cfg = OFDMConfig()
    r = 0.5
    k = compute_k(cfg, r)
    test_bits = np.random.randint(0, 2, k)
    cbs = ldpc_encode(test_bits, cfg, r)
    noisy_llrs = [1 - 2 * cb for cb in cbs]
    dec = ldpc_decode(noisy_llrs, cfg, r)
    print(f"比特是否一致: {np.all(dec == test_bits)}")
