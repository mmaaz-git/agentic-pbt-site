import numpy as np
from scipy.signal.windows import tukey

# Test with extremely small alpha values
print("Testing tukey function with tiny alpha values:")
print("-" * 50)

# Test case 1: M=2, alpha=1e-309
w = tukey(2, alpha=1e-309, sym=True)
print(f"tukey(2, alpha=1e-309) = {w}")
print(f"Contains NaN: {np.any(np.isnan(w))}")
print()

# Test case 2: M=10, alpha=1e-309
w = tukey(10, alpha=1e-309, sym=True)
print(f"tukey(10, alpha=1e-309) = {w}")
print(f"Contains NaN: {np.any(np.isnan(w))}")
print()

# Test case 3: Even smaller alpha - 1e-320
w = tukey(2, alpha=1e-320, sym=True)
print(f"tukey(2, alpha=1e-320) = {w}")
print(f"Contains NaN: {np.any(np.isnan(w))}")
print()

# Test boundary cases to find where it starts failing
print("Finding the boundary where NaN appears:")
print("-" * 50)
test_alphas = [1e-50, 1e-100, 1e-200, 1e-300, 1e-305, 1e-308, 1e-309, 1e-310, 1e-320]
for alpha in test_alphas:
    w = tukey(2, alpha=alpha, sym=True)
    has_nan = np.any(np.isnan(w))
    print(f"alpha={alpha:e}: Contains NaN = {has_nan}")