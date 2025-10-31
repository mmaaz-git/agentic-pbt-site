import numpy as np
from scipy.signal import windows

# Test several window sizes
for M in [2, 3, 4, 5, 10, 100]:
    for sym in [True, False]:
        window = windows.flattop(M, sym=sym)
        max_val = np.max(window)
        print(f"flattop({M:3d}, sym={sym!s:5s}): max value = {max_val:.15f}, exceeds 1.0? {max_val > 1.0}")

# Check using get_window as in the hypothesis test
window = windows.get_window('flattop', 2, fftbins=True)
max_val = np.max(np.abs(window))
print(f"\nget_window('flattop', 2, fftbins=True): max value = {max_val:.15f}, exceeds 1.0? {max_val > 1.0}")