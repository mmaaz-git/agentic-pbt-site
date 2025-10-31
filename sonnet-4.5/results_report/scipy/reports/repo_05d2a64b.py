import numpy as np
from scipy import stats

# Create test data
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

print("Testing scipy.stats.quantile with integer p values:")
print("=" * 60)
print("Test data x =", x)
print()

# Test 1: Integer 0 (should fail)
print("Test 1: stats.quantile(x, 0) with integer 0")
print("-" * 40)
try:
    result = stats.quantile(x, 0)
    print(f"Success: {result}")
except ValueError as e:
    print(f"ValueError: {e}")

print()

# Test 2: Float 0.0 (should work)
print("Test 2: stats.quantile(x, 0.0) with float 0.0")
print("-" * 40)
try:
    result = stats.quantile(x, 0.0)
    print(f"Success: {result}")
except ValueError as e:
    print(f"ValueError: {e}")

print()

# Test 3: Integer 1 (should fail)
print("Test 3: stats.quantile(x, 1) with integer 1")
print("-" * 40)
try:
    result = stats.quantile(x, 1)
    print(f"Success: {result}")
except ValueError as e:
    print(f"ValueError: {e}")

print()

# Test 4: Float 1.0 (should work)
print("Test 4: stats.quantile(x, 1.0) with float 1.0")
print("-" * 40)
try:
    result = stats.quantile(x, 1.0)
    print(f"Success: {result}")
except ValueError as e:
    print(f"ValueError: {e}")

print()

# Compare with NumPy's behavior
print("Comparison with NumPy functions:")
print("=" * 60)

# Test NumPy quantile
print("np.quantile(x, 0) =", np.quantile(x, 0))
print("np.quantile(x, 1) =", np.quantile(x, 1))
print("np.percentile(x, 0) =", np.percentile(x, 0))
print("np.percentile(x, 100) =", np.percentile(x, 100))

print()
print("Conclusion: NumPy accepts integer p values without issue,")
print("while scipy.stats.quantile unnecessarily rejects them.")