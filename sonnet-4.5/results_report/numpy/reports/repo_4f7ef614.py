import numpy as np
import numpy.ma as ma

print("Testing numpy.ma.compress_rows and compress_cols shape inconsistency")
print("=" * 70)

# Test case 1: Fully masked 2-D array
print("\nTest 1: Fully masked 2x3 array")
print("-" * 40)
data = np.array([[1., 2., 3.],
                 [4., 5., 6.]])
mask = np.ones((2, 3), dtype=bool)
arr = ma.array(data, mask=mask)

print(f"Input shape: {arr.shape}")
print(f"Input array:\n{arr}")
print(f"Input mask:\n{arr.mask}")

result_rows = ma.compress_rows(arr)
print(f"\ncompress_rows result shape: {result_rows.shape}")
print(f"compress_rows result: {result_rows}")
print(f"Expected shape: (0, 3)")
print(f"Result dimensionality: {result_rows.ndim}")

result_cols = ma.compress_cols(arr)
print(f"\ncompress_cols result shape: {result_cols.shape}")
print(f"compress_cols result: {result_cols}")
print(f"Expected shape: (2, 0)")
print(f"Result dimensionality: {result_cols.ndim}")

# Test case 2: Partially masked array for comparison
print("\n\nTest 2: Partially masked 2x3 array (for comparison)")
print("-" * 40)
data_partial = np.array([[1., 2., 3.],
                        [4., 5., 6.]])
mask_partial = np.array([[True, False, False],
                        [False, True, False]])
arr_partial = ma.array(data_partial, mask=mask_partial)

print(f"Input shape: {arr_partial.shape}")
print(f"Input array:\n{arr_partial}")
print(f"Input mask:\n{arr_partial.mask}")

result_rows_partial = ma.compress_rows(arr_partial)
print(f"\ncompress_rows result shape: {result_rows_partial.shape}")
print(f"compress_rows result:\n{result_rows_partial}")
print(f"Result dimensionality: {result_rows_partial.ndim}")

result_cols_partial = ma.compress_cols(arr_partial)
print(f"\ncompress_cols result shape: {result_cols_partial.shape}")
print(f"compress_cols result:\n{result_cols_partial}")
print(f"Result dimensionality: {result_cols_partial.ndim}")

# Demonstrate downstream error
print("\n\nTest 3: Demonstrating downstream failure")
print("-" * 40)
print("Attempting to access shape[1] on fully masked result:")
try:
    fully_masked_result = ma.compress_rows(arr)
    print(f"Result shape: {fully_masked_result.shape}")
    print(f"Accessing shape[1]: {fully_masked_result.shape[1]}")
except IndexError as e:
    print(f"ERROR: IndexError occurred - {e}")
    print("This would break code expecting a 2-D array!")