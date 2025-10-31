import numpy as np
import scipy.signal.windows as windows

M = 3
beta = 710.0
window = windows.kaiser(M, beta)

print(f"windows.kaiser({M}, {beta}) = {window}")
print(f"Contains NaN: {np.any(np.isnan(window))}")
print(f"All finite: {np.all(np.isfinite(window))}")

assert np.all(np.isfinite(window)), "Kaiser window should not contain NaN values"