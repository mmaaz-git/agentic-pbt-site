import numpy as np
from scipy.cluster.vq import whiten
import warnings

# Test 1: Truly constant column (small value)
print("=== Test 1: Small constant values ===")
obs1 = np.array([[1.0, 1.0]] * 10)
print(f"Input: {obs1[0]}")
print(f"Std dev: {np.std(obs1, axis=0)}")
print(f"Std == 0: {np.std(obs1, axis=0) == 0}")

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    result1 = whiten(obs1)
    if w:
        print(f"Warning raised: {w[0].message}")
    else:
        print("No warning raised")

print(f"Output: {result1[0]}")
print()

# Test 2: Truly constant column (large value, like in bug report)
print("=== Test 2: Large constant values ===")
obs2 = np.array([[93206.82233024, 93206.82233024]] * 40)
print(f"Input: {obs2[0]}")
print(f"Std dev: {np.std(obs2, axis=0)}")
print(f"Std == 0: {np.std(obs2, axis=0) == 0}")

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    result2 = whiten(obs2)
    if w:
        print(f"Warning raised: {w[0].message}")
    else:
        print("No warning raised")

print(f"Output: {result2[0]}")
print()

# Test 3: Manually create a perfect constant array
print("=== Test 3: Explicitly created constant array ===")
obs3 = np.ones((10, 2)) * 42.0
print(f"Input: {obs3[0]}")
print(f"Std dev: {np.std(obs3, axis=0)}")
print(f"Std == 0: {np.std(obs3, axis=0) == 0}")

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    result3 = whiten(obs3)
    if w:
        print(f"Warning raised: {w[0].message}")
    else:
        print("No warning raised")

print(f"Output: {result3[0]}")
print()

# Test 4: Check if issue is specific to how the array is created
print("=== Test 4: Different array creation methods ===")
# Method A: Using list multiplication (as in bug report)
val = 93206.82233024
obs4a = np.array([[val, val]] * 40)
print(f"Method A (list multiplication):")
print(f"  Std dev: {np.std(obs4a, axis=0)}")
print(f"  Std == 0: {np.std(obs4a, axis=0) == 0}")

# Method B: Using np.full
obs4b = np.full((40, 2), val)
print(f"Method B (np.full):")
print(f"  Std dev: {np.std(obs4b, axis=0)}")
print(f"  Std == 0: {np.std(obs4b, axis=0) == 0}")

# Method C: Using np.ones
obs4c = np.ones((40, 2)) * val
print(f"Method C (np.ones * val):")
print(f"  Std dev: {np.std(obs4c, axis=0)}")
print(f"  Std == 0: {np.std(obs4c, axis=0) == 0}")