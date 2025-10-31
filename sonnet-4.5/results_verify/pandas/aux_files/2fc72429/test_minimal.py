import numpy as np
import scipy.signal.windows as w

M = 2
alpha = 1e-300

window = w.tukey(M, alpha, sym=True)
print(f"tukey({M}, {alpha}, sym=True) = {window}")
print(f"Reversed: {window[::-1]}")
print(f"Symmetric: {np.allclose(window, window[::-1])}")

# Also test with the value found by hypothesis
M2 = 2
alpha2 = 2.8429516001933894e-89

window2 = w.tukey(M2, alpha2, sym=True)
print(f"\ntukey({M2}, {alpha2}, sym=True) = {window2}")
print(f"Reversed: {window2[::-1]}")
print(f"Symmetric: {np.allclose(window2, window2[::-1])}")