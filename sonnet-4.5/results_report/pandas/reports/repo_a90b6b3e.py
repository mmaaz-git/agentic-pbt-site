import numpy as np
import numpy.ma as ma

# Test 1: Fully masked array
arr1 = ma.array([[99]], mask=[[True]])
result1 = ma.compress_rowcols(arr1)
print(f"compress_rowcols fully masked: shape={arr1.shape} -> {result1.shape}, ndim={arr1.ndim} -> {result1.ndim}")

result2 = ma.compress_rows(arr1)
print(f"compress_rows fully masked: shape={arr1.shape} -> {result2.shape}, ndim={arr1.ndim} -> {result2.ndim}")

result3 = ma.compress_cols(arr1)
print(f"compress_cols fully masked: shape={arr1.shape} -> {result3.shape}, ndim={arr1.ndim} -> {result3.ndim}")

# Test 2: Partially masked 2D array where all rows/columns get removed
arr2 = ma.array([[1, 2], [3, 4]], mask=[[True, False], [False, True]])
result4 = ma.compress_rowcols(arr2)
print(f"\nPartially masked, all removed: shape={arr2.shape} -> {result4.shape}, ndim={arr2.ndim} -> {result4.ndim}")

# Test 3: Show the actual array results
print(f"\nActual results:")
print(f"Fully masked [[--]]: result = {result1}, type = {type(result1)}")
print(f"Partially masked with all removed: result = {result4}, type = {type(result4)}")