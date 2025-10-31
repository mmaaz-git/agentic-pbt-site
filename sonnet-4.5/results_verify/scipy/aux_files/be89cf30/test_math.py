import numpy as np

# Test what happens with the problematic expression
alpha = 5e-324
M = 2

print(f"alpha = {alpha}")
print(f"Is alpha positive? {alpha > 0}")
print(f"Is alpha == 0? {alpha == 0}")
print(f"alpha <= 0? {alpha <= 0}")

# The problematic expression from line 946
print(f"\n-2.0/alpha = {-2.0/alpha}")

# Test cos of infinity
print(f"np.cos(np.inf) = {np.cos(np.inf)}")
print(f"np.cos(-np.inf) = {np.cos(-np.inf)}")

# The full problematic expression for n3[0] when M=2
n3_val = 1  # For M=2, n3 = [1]
expr = -2.0/alpha + 1 + 2.0*n3_val/alpha/(M-1)
print(f"\nFull expression inside cos: {expr}")
print(f"np.pi * expr = {np.pi * expr}")
print(f"np.cos(np.pi * expr) = {np.cos(np.pi * expr)}")

# Test with progressively smaller alpha values
print("\n\nTesting division by very small alphas:")
test_alphas = [1e-10, 1e-50, 1e-100, 1e-200, 1e-300, 1e-320, 5e-324]
for a in test_alphas:
    try:
        result = -2.0/a
        print(f"alpha={a:.2e}: -2.0/alpha = {result}")
    except:
        print(f"alpha={a:.2e}: Division failed")

# Check numpy/python's smallest positive float
import sys
print(f"\n\nSmallest positive float: {sys.float_info.min}")
print(f"5e-324 compared to min: {5e-324 / sys.float_info.min}")
print(f"Is 5e-324 a subnormal number? {5e-324 < sys.float_info.min}")