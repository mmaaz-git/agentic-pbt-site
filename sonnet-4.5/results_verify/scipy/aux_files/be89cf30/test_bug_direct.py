import numpy as np
from scipy.signal import windows

M = 2
alpha = 5e-324

w = windows.tukey(M, alpha)
print(f"tukey({M}, {alpha}) = {w}")
print(f"Contains NaN: {np.any(np.isnan(w))}")
print(f"Window values: {w}")
print(f"Are all values >= 0? {np.all(~np.isnan(w) & (w >= 0))}")