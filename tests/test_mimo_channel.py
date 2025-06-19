import numpy as np
import sys
from pathlib import Path

# Ensure project root is on the Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.channel import awgn_channel, rayleigh_channel


def test_awgn_2x2_shape_and_power():
    num_tx = num_rx = 2
    N = 1000
    tx = (np.random.randn(num_tx, N) + 1j * np.random.randn(num_tx, N)) / np.sqrt(2)
    rx = awgn_channel(tx, num_rx=num_rx, num_tx=num_tx)
    assert rx.shape == (num_rx, N)
    noise = rx - tx
    assert np.allclose(np.mean(np.abs(noise)**2), 1.0, atol=0.2)


def test_rayleigh_2x2_block_fading():
    num_tx = num_rx = 2
    N = 1000
    tx = (np.random.randn(num_tx, N) + 1j * np.random.randn(num_tx, N)) / np.sqrt(2)
    rx, h = rayleigh_channel(tx, block_fading=True, num_rx=num_rx, num_tx=num_tx)
    assert rx.shape == (num_rx, N)
    assert h.shape == (num_rx, num_tx)
    power = np.mean(np.abs(h)**2)
    assert 0.2 < power < 2.0

