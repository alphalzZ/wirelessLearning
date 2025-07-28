# 📄 文档 1 —— Python OFDM 仿真开发指南（功能 & 流程篇）

> **目标**：提供一份流程清晰、功能完整的开发蓝图，指导 AI 编码助手（或工程师本人）快速实现一条可扩展的基带 OFDM 仿真链路。

## 1. 功能范围与里程碑

| 里程碑 | 必选功能                                                         | 交付物                           |
| ------ | ---------------------------------------------------------------- | -------------------------------- |
| M1     | 参数配置模块<br>位映射 → QAM 调制<br>IFFT 加循环前缀              | `ofdm_tx.py` + `config.yaml`     |
| M2     | AWGN + 随机多径信道（可选 Rayleigh）<br>去除 CP → FFT → 信道均衡 | `channel.py` + `ofdm_rx.py`      |
| M3     | BER / SER 统计、SNR 扫描、批量仿真脚本                           | `metrics.py` + `run_experiments.py` |
| M4     | 可选：帧同步、导频估计、OFDM‑MIMO 骨架                             | `sync.py` / `mimo.py`（占位）     |

## 2. 关键系统参数（可在 `config.yaml` 中集中维护）

| 参数           | 典型默认值 | 说明                               |
| -------------- | ---------- | ---------------------------------- |
| `n_fft`        | 64         | IFFT/FFT 点数                     |
| `cp_len`       | 16         | 循环前缀长度                       |
| `mod_order`    | 4          | QPSK (2 bits/sym)，可设 16/64-QAM  |
| `num_symbols`  | 10 000     | 单次仿真的 OFDM 符号数             |
| `snr_db_list`  | 0‑30 dB    | 多 SNR 扫描                        |
| `channel_type` | `awgn`     | 可选 `awgn` / `rayleigh` / `tapped_delay` |

## 3. 顶层流程（Tx → Channel → Rx）

```mermaid
graph LR
  A(Bitstream) --> B[QAM Mod]
  B --> C[IFFT]
  C --> D[Add CP]
  D --> E(Channel)
  E --> F[Remove CP]
  F --> G[FFT]
  G --> H[Equalizer]
  H --> I[QAM Demod]
  I --> J[BER/SER Calc]
```

1. **发端**  
   1.1 伪随机比特流 ➜ Gray 映射 ➜ QAM 点  
   1.2 IFFT 形成时域符号；前端首尾拼接循环前缀  

2. **信道**  
   - AWGN：向量级随机噪声  
   - 多径：卷积实现；或利用 `numpy.fft` 频域乘法  

3. **接收端**  
   3.1 删除 CP；FFT 还原到频域  
   3.2 零强制 or MMSE 均衡（Rayleigh 时需要）  
   3.3 判决 → 逆 Gray → 还原比特  

4. **指标**  
   - 逐符号 BER / SER  
   - 支持批处理 / 多 SNR 曲线  

## 4. 目录建议

```
ofdm_sim/
├── src/
│   ├── config.py           # 参数解析与 dataclass
│   ├── ofdm_tx.py          # 发送链路
│   ├── ofdm_rx.py          # 接收链路
│   ├── channel.py          # 信道模型
│   ├── metrics.py          # 统计工具
│   └── utils.py            # 公共函数（e.g. QAM）
├── experiments/
│   └── run_experiments.py  # 脚本入口
├── tests/                  # pytest 单元测试
└── README.md               # 项目说明（可放本文档）
```

## 5. 快速开始（示范代码片段）

```python
# run_experiments.py
from src.config import cfg
from src.ofdm_tx import ofdm_tx
from src.channel import channel
from src.ofdm_rx import ofdm_rx
from src.metrics import ber
import numpy as np

def sim_once(snr_db: float) -> float:
    bits_tx = np.random.randint(0, 2, cfg.num_bits)
    tx_sig  = ofdm_tx(bits_tx, cfg)
    rx_sig  = channel(tx_sig, snr_db, cfg)
    bits_rx = ofdm_rx(rx_sig, cfg)
    return ber(bits_tx, bits_rx)

if __name__ == "__main__":
    for snr in cfg.snr_db_list:
        print(f"SNR={snr:2} dB → BER={sim_once(snr):.3e}")
```

## 6. 验证与扩展

- **单元测试**：  
  - `test_qam.py`：映射 & 逆映射自反性  
  - `test_ofdm_loopback.py`：无噪声、理想信道 BER=0  
- **数值交叉**：与 MATLAB or GNU Radio 结果比对  
- **后续可加**：  
  - 导频‑LS/DFT‑LS 信道估计  
  - PAPR 统计、Clipping/Windowing 降峰  
  - LDPC / Polar 外层编码
