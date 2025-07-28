import numpy as np
import sys
from pathlib import Path

# Ensure project root is on the Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.ofdm_tx import qam_modulation
from src.ofdm_rx import qam_demodulation

def test_llr_matches_bits_64qam():
    Qm = 6
    bits = np.random.randint(0, 2, Qm * 100)
    syms = qam_modulation(bits, Qm)
    llr = qam_demodulation(syms, Qm, return_llr=True, noise_var=1e-12)
    expected = np.where(bits == 0, 1, -1)
    assert np.all(np.sign(llr) == expected)
