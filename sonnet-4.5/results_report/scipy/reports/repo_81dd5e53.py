import numpy as np
import scipy.signal.windows as windows

# Test odd values of M
print("Testing odd values of M:")
for M in [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]:
    w = windows.flattop(M)
    max_val = np.max(w)
    print(f"flattop({M:2}): max = {max_val:.15f}, exceeds 1.0: {max_val > 1.0}")

print("\nTesting even values of M:")
for M in [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]:
    w = windows.flattop(M)
    max_val = np.max(w)
    print(f"flattop({M:2}): max = {max_val:.15f}, exceeds 1.0: {max_val > 1.0}")

print("\nVerifying coefficient sum:")
coeffs = [0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368]
coeff_sum = sum(coeffs)
print(f"Sum of coefficients: {coeff_sum:.15f}")
print(f"Excess over 1.0: {coeff_sum - 1.0:.15e}")