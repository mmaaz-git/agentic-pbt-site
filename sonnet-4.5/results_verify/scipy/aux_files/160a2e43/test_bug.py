import numpy as np
from scipy.signal import windows

window = windows.flattop(3, sym=True)
max_val = np.max(window)

print(f"flattop(3, sym=True) = {window}")
print(f"max value = {max_val:.15f}")
print(f"exceeds 1.0? {max_val > 1.0}")

assert max_val <= 1.0, f"Maximum value {max_val} exceeds documented limit of 1.0"