import numpy as np
import scipy.signal.windows as w

M = 2
alpha = 1e-300

window = w.tukey(M, alpha, sym=True)
print(f"tukey({M}, {alpha}, sym=True) = {window}")
print(f"Reversed: {window[::-1]}")
print(f"Symmetric: {np.allclose(window, window[::-1])}")
print(f"Are values equal? {window[0]} == {window[-1]}: {window[0] == window[-1]}")