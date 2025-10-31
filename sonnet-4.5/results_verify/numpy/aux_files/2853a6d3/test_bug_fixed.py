import numpy as np
import numpy.ma as ma

print("=" * 50)
print("Testing the bug reproduction...")
print("=" * 50)

arr = np.ones((3, 4))
all_masked = ma.array(arr, mask=np.ones((3, 4), dtype=bool))

result = ma.compress_rowcols(all_masked, axis=0)

print(f"Input shape: {all_masked.shape}")
print(f"Result shape: {result.shape}")
print(f"Result ndim: {result.ndim}")
print(f"Result dtype: {result.dtype}")
print(f"Result: {result}")

try:
    assert result.ndim == 2, f"Expected 2D, got {result.ndim}D"
    print("Assertion passed: Result is 2D")
except AssertionError as e:
    print(f"AssertionError: {e}")

print("\n" + "=" * 50)
print("Testing with partially masked array...")
print("=" * 50)

# Test with partially masked array
arr2 = np.ones((3, 4))
partial_mask = np.array([[True, True, True, True],
                         [False, False, False, False],
                         [True, True, True, True]])
partial_masked = ma.array(arr2, mask=partial_mask)

result2 = ma.compress_rowcols(partial_masked, axis=0)
print(f"Partially masked input shape: {partial_masked.shape}")
print(f"Partially masked result shape: {result2.shape}")
print(f"Partially masked result ndim: {result2.ndim}")
print(f"Partially masked result: {result2}")

print("\n" + "=" * 50)
print("Testing axis=1 (columns)...")
print("=" * 50)

# Test axis=1 with all masked
result3 = ma.compress_rowcols(all_masked, axis=1)
print(f"All masked, axis=1 result shape: {result3.shape}")
print(f"All masked, axis=1 result ndim: {result3.ndim}")

# Test axis=1 with partial mask
partial_mask_cols = np.array([[True, False, True, False],
                              [True, False, True, False],
                              [True, False, True, False]])
partial_masked_cols = ma.array(arr2, mask=partial_mask_cols)
result4 = ma.compress_rowcols(partial_masked_cols, axis=1)
print(f"Partially masked, axis=1 input shape: {partial_masked_cols.shape}")
print(f"Partially masked, axis=1 result shape: {result4.shape}")
print(f"Partially masked, axis=1 result ndim: {result4.ndim}")

print("\n" + "=" * 50)
print("Testing axis=None (both rows and columns)...")
print("=" * 50)

result5 = ma.compress_rowcols(all_masked, axis=None)
print(f"All masked, axis=None result shape: {result5.shape}")
print(f"All masked, axis=None result ndim: {result5.ndim}")

print("\n" + "=" * 50)
print("Additional tests for consistency...")
print("=" * 50)

# Test what happens with no masked values
no_mask_arr = ma.array(np.ones((3, 4)), mask=False)
result6 = ma.compress_rowcols(no_mask_arr, axis=0)
print(f"No mask, axis=0 result shape: {result6.shape}")
print(f"No mask, axis=0 result ndim: {result6.ndim}")

# Test with single row remaining
single_row_mask = np.array([[True, True, True, True],
                            [False, False, False, False],
                            [True, True, True, True]])
single_row = ma.array(np.arange(12).reshape(3, 4), mask=single_row_mask)
result7 = ma.compress_rowcols(single_row, axis=0)
print(f"Single row remaining result shape: {result7.shape}")
print(f"Single row remaining result ndim: {result7.ndim}")
print(f"Single row remaining result: {result7}")

# Check regular numpy behavior with empty arrays
print("\n" + "=" * 50)
print("Regular numpy empty array behavior...")
print("=" * 50)

empty_2d = np.array([]).reshape(0, 4)
print(f"Empty 2D array shape (0, 4): {empty_2d.shape}")
print(f"Empty 2D array ndim: {empty_2d.ndim}")

empty_1d = np.array([])
print(f"Empty 1D array shape: {empty_1d.shape}")
print(f"Empty 1D array ndim: {empty_1d.ndim}")