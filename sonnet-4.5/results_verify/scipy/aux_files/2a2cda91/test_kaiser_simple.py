import numpy as np
import scipy.signal.windows as windows

M = 3
beta = 710.0
window = windows.kaiser(M, beta)

print(f"windows.kaiser({M}, {beta}) = {window}")
print(f"Window contains NaN: {np.any(np.isnan(window))}")
print(f"Window contains Inf: {np.any(np.isinf(window))}")
print(f"All values finite: {np.all(np.isfinite(window))}")

assert np.all(np.isfinite(window)), "Kaiser window should not contain NaN values"