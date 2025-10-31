import numpy as np
import numpy.ma as ma

print("Testing with all elements masked:")
data = np.array([[1., 2., 3.],
                 [4., 5., 6.]])
mask = np.ones((2, 3), dtype=bool)
arr = ma.array(data, mask=mask)

result_rows = ma.compress_rows(arr)
print(f"compress_rows shape: {result_rows.shape}")
print(f"Expected: (0, 3), Actual: {result_rows.shape}")
print(f"Result ndim: {result_rows.ndim}")
print(f"Result: {result_rows}")
print()

result_cols = ma.compress_cols(arr)
print(f"compress_cols shape: {result_cols.shape}")
print(f"Expected: (2, 0), Actual: {result_cols.shape}")
print(f"Result ndim: {result_cols.ndim}")
print(f"Result: {result_cols}")
print()

print("Testing with partial mask:")
data_partial = np.array([[1., 2., 3.],
                        [4., 5., 6.]])
mask_partial = np.array([[True, False, False],
                        [False, True, False]])
arr_partial = ma.array(data_partial, mask=mask_partial)

result_rows_partial = ma.compress_rows(arr_partial)
print(f"compress_rows (partial mask) shape: {result_rows_partial.shape}")
print(f"Result ndim: {result_rows_partial.ndim}")

result_cols_partial = ma.compress_cols(arr_partial)
print(f"compress_cols (partial mask) shape: {result_cols_partial.shape}")
print(f"Result ndim: {result_cols_partial.ndim}")