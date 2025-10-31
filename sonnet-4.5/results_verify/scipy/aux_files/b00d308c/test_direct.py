import scipy.stats
import numpy as np

n = 3
p = 1.1125369292536007e-308

print(f"Testing with n={n}, p={p}")
print(f"p is a valid float: {isinstance(p, float)}")
print(f"p is in valid range [0,1]: {0 <= p <= 1}")
print(f"p value: {p}")

try:
    result = scipy.stats.binom.pmf(0, n, p)
    print(f"Result: {result}")
except OverflowError as e:
    print(f"OverflowError: {e}")

# Try with different values to understand boundaries
print("\nTesting with different values:")

# Test with p=1e-300
try:
    p_test = 1e-300
    result = scipy.stats.binom.pmf(0, 3, p_test)
    print(f"p=1e-300, n=3, pmf(0): {result}")
except OverflowError as e:
    print(f"p=1e-300 failed: {e}")

# Test with p=1e-307
try:
    p_test = 1e-307
    result = scipy.stats.binom.pmf(0, 3, p_test)
    print(f"p=1e-307, n=3, pmf(0): {result}")
except OverflowError as e:
    print(f"p=1e-307 failed: {e}")

# Test with n=1 and n=2 with the problematic p
try:
    result = scipy.stats.binom.pmf(0, 1, p)
    print(f"n=1, p={p}, pmf(0): {result}")
except OverflowError as e:
    print(f"n=1 failed: {e}")

try:
    result = scipy.stats.binom.pmf(0, 2, p)
    print(f"n=2, p={p}, pmf(0): {result}")
except OverflowError as e:
    print(f"n=2 failed: {e}")

# Check mathematical expectation
print(f"\nMathematical expectation: (1-p)^n = (1-{p})^{n} ≈ {(1-p)**n}")