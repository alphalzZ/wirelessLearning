import numpy as np

# ---------- 系统参数 ----------
N_re  = 64         # 每符号导频 RE 数
R_rx  = 2          # 接收天线数
Es    = 1          # 导频功率
snr_dB = 20

# ---------- 1. 生成发射导频 ----------
n = np.arange(N_re)
a = np.exp(-1j * np.pi * n * (n + 1) / N_re) * np.sqrt(Es)   # ZC-root=1
c0 = np.ones (N_re, dtype=np.float32)                        # [+1,+1,...]
c1 = np.where(n % 2, -1, 1).astype(np.float32)               # [+1,-1,+1,-1]

# ---------- 2. 随机信道 & 噪声 ----------
h0 = (np.random.randn(R_rx) + 1j*np.random.randn(R_rx)) / np.sqrt(2)
h1 = (np.random.randn(R_rx) + 1j*np.random.randn(R_rx)) / np.sqrt(2)
awgn = (np.random.randn(N_re, R_rx) + 1j*np.random.randn(N_re, R_rx))
awgn *= np.sqrt(Es / (2 * 10**(snr_dB/10)))

# ---------- 3. 发射 & 接收 ----------
tx0 = a * c0                    # Tx 天线 0
tx1 = a * c1                    # Tx 天线 1
y   = np.outer(tx0, h0) + np.outer(tx1, h1) + awgn   # shape (N_re, R_rx)

# ---------- 4. OCC 解码 ----------
# 4-1 去调制
y_despread = y / a[:, None]               # element-wise

# 4-2 相关分离   (偶=0,2,..., 奇=1,3,...)
even = y_despread[::2]
odd  = y_despread[1::2]
h0_hat = (even + odd).mean(axis=0) / 2             # shape (R_rx,)
h1_hat = (even - odd).mean(axis=0) / 2

print("真实 h0:", h0, "\n估计 h0:", h0_hat)
print("真实 h1:", h1, "\n估计 h1:", h1_hat)
