import numpy as np
import scipy.signal.windows as w

# Test with normal alpha values to see expected behavior
print("Testing with normal alpha values:")
for alpha in [0.5, 0.1, 0.01, 1e-10, 1e-50, 1e-100, 1e-200, 1e-300]:
    window = w.tukey(2, alpha, sym=True)
    is_symmetric = np.allclose(window, window[::-1])
    print(f"alpha={alpha:e}: window={window}, symmetric={is_symmetric}")

print("\nTesting with M=3:")
for alpha in [0.5, 1e-100, 1e-300]:
    window = w.tukey(3, alpha, sym=True)
    is_symmetric = np.allclose(window, window[::-1])
    print(f"alpha={alpha:e}: window={window}, symmetric={is_symmetric}")

print("\nTesting symmetry property for larger windows:")
M = 10
alpha = 1e-300
window = w.tukey(M, alpha, sym=True)
is_symmetric = np.allclose(window, window[::-1])
print(f"M={M}, alpha={alpha:e}: symmetric={is_symmetric}")
if not is_symmetric:
    print(f"First half: {window[:M//2]}")
    print(f"Last half reversed: {window[M//2:][::-1]}")