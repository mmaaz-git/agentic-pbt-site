import numpy as np

# Understanding floating point precision issue
val = 93206.82233024

# Create array where all values should be identical
arr = np.array([[val, val]] * 40)

print("=== Floating Point Precision Analysis ===")
print(f"Value: {val}")
print(f"Value in hex: {val.hex()}")
print()

# Check if all values are truly identical
print("Are all values bit-identical?")
first = arr[0, 0]
for i in range(arr.shape[0]):
    for j in range(arr.shape[1]):
        if arr[i, j] != first:
            print(f"  Found difference at [{i}, {j}]: {arr[i, j]} != {first}")
            print(f"  Difference: {arr[i, j] - first}")

all_equal = np.all(arr == first)
print(f"  All equal via numpy: {all_equal}")
print()

# Manual std calculation to see what's happening
print("Manual standard deviation calculation:")
mean = np.mean(arr[:, 0])
print(f"Mean: {mean}")
print(f"Mean - first value: {mean - first}")

# Compute variance manually
variance = np.mean((arr[:, 0] - mean) ** 2)
print(f"Variance: {variance}")
std_manual = np.sqrt(variance)
print(f"Std (manual): {std_manual}")
print(f"Std (numpy): {np.std(arr[:, 0])}")
print()

# Check if this is a representation issue
print("Checking representation precision:")
print(f"First element exact: {arr[0, 0]}")
print(f"Mean exact: {mean}")
print(f"Are they exactly equal? {arr[0, 0] == mean}")
print(f"Difference: {mean - arr[0, 0]}")

# What happens with smaller values?
print("\n=== Comparison with smaller value ===")
small_val = 93.20682233024  # Same digits, smaller magnitude
small_arr = np.array([[small_val, small_val]] * 40)
print(f"Small value: {small_val}")
print(f"Small array std: {np.std(small_arr, axis=0)}")
print(f"Small std == 0: {np.std(small_arr, axis=0) == 0}")

# Even smaller
tiny_val = 0.9320682233024
tiny_arr = np.array([[tiny_val, tiny_val]] * 40)
print(f"Tiny value: {tiny_val}")
print(f"Tiny array std: {np.std(tiny_arr, axis=0)}")
print(f"Tiny std == 0: {np.std(tiny_arr, axis=0) == 0}")

# Let's see the actual computation that causes the issue
print("\n=== Understanding the division ===")
std = np.std(arr, axis=0)[0]
print(f"Original value: {val}")
print(f"Std dev: {std}")
print(f"Result of division: {val / std}")
print(f"Expected if std was treated as 0: {val} (unchanged)")

# Check if this matches the bug report's multiplication factor
expected_factor = 17179869184.0  # From bug report
actual_factor = val / std if std != 0 else 1
print(f"\nMultiplication factor from bug report: {expected_factor}")
print(f"Our calculated factor: {actual_factor}")
print(f"Match? {abs(actual_factor - expected_factor) < 1}")