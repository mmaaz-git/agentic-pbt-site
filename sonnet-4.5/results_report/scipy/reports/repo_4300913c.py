import numpy as np
import scipy.ndimage as ndi

# Create a constant array filled with 5.0
arr = np.full((10, 10), 5.0, dtype=np.float64)

# Apply Sobel filter with mode='constant' and cval=5.0
result = ndi.sobel(arr, mode='constant', cval=5.0)

print("Input: 10x10 constant array filled with 5.0")
print(f"mode='constant', cval=5.0")
print(f"\nResult array:\n{result}")

print(f"\nExpected: All zeros (gradient of constant is zero)")
print(f"Actual:")
print(f"  Top row = {result[0, :]}")
print(f"  Bottom row = {result[-1, :]}")
print(f"  Left column = {result[:, 0]}")
print(f"  Right column = {result[:, -1]}")
print(f"  Max absolute value: {np.max(np.abs(result))}")
print(f"  Are all values zero? {np.allclose(result, 0.0, atol=1e-10)}")

# Compare with mode='nearest' which should work correctly
print("\n--- Comparison with mode='nearest' ---")
result_nearest = ndi.sobel(arr, mode='nearest')
print(f"With mode='nearest': max absolute value = {np.max(np.abs(result_nearest)):.2e}")
print(f"Are all values zero? {np.allclose(result_nearest, 0.0, atol=1e-10)}")

# Show the issue is at boundaries
print("\n--- Analysis of non-zero locations ---")
non_zero_mask = np.abs(result) > 1e-10
print(f"Non-zero locations (True means non-zero):")
print(non_zero_mask.astype(int))
print(f"Total non-zero elements: {np.sum(non_zero_mask)} out of {result.size}")