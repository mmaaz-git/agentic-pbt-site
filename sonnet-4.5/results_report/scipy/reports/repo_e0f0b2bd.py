import numpy as np
from scipy.signal import windows

# Test case that demonstrates the bug
window = windows.flattop(3, sym=True)
max_val = np.max(window)

print(f"flattop(3, sym=True) = {window}")
print(f"max value = {max_val:.15f}")
print(f"exceeds 1.0? {max_val > 1.0}")

# Additional test cases
print("\nAdditional test cases:")
for M in [2, 3, 4, 5, 10]:
    for sym in [True, False]:
        window = windows.flattop(M, sym=sym)
        max_val = np.max(window)
        if max_val > 1.0:
            print(f"  M={M}, sym={sym}: max = {max_val:.15f} (exceeds 1.0)")

# Verify coefficients sum
coefficients = [0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368]
coeff_sum = sum(coefficients)
print(f"\nCoefficients sum = {coeff_sum:.15f}")
print(f"Exceeds 1.0 by: {coeff_sum - 1.0:.15e}")

# This should fail according to the documentation
assert max_val <= 1.0, f"Maximum value {max_val} exceeds documented limit of 1.0"