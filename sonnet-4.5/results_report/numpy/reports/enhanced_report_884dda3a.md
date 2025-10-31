# Bug Report: numpy.ma compress_rows/compress_cols/compress_rowcols Loses 2D Structure on Empty Results

**Target**: `numpy.ma.compress_rows`, `numpy.ma.compress_cols`, `numpy.ma.compress_rowcols`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The compress_rows, compress_cols, and compress_rowcols functions incorrectly return 1D arrays when all rows/columns are masked, instead of maintaining the expected 2D structure.

## Property-Based Test

```python
import numpy as np
import numpy.ma as ma
from hypothesis import given, strategies as st, settings
from hypothesis.extra import numpy as npst

@st.composite
def masked_2d_arrays(draw):
    rows = draw(st.integers(min_value=1, max_value=5))
    cols = draw(st.integers(min_value=1, max_value=5))
    data = draw(npst.arrays(dtype=np.float64, shape=(rows, cols),
                           elements=st.floats(allow_nan=False, allow_infinity=False,
                                            min_value=-100, max_value=100)))
    mask = draw(npst.arrays(dtype=bool, shape=(rows, cols)))
    return ma.array(data, mask=mask)

@given(masked_2d_arrays())
@settings(max_examples=500)
def test_compress_rows_preserves_2d_structure(arr):
    result = ma.compress_rows(arr)
    assert result.ndim == 2

@given(masked_2d_arrays())
@settings(max_examples=500)
def test_compress_cols_preserves_2d_structure(arr):
    result = ma.compress_cols(arr)
    assert result.ndim == 2

@given(masked_2d_arrays())
@settings(max_examples=500)
def test_compress_rowcols_preserves_2d_structure(arr):
    result = ma.compress_rowcols(arr)
    assert result.ndim == 2

if __name__ == "__main__":
    # Run the test for compress_rows
    print("Testing compress_rows preserves 2D structure...")
    try:
        test_compress_rows_preserves_2d_structure()
        print("Test passed!")
    except AssertionError as e:
        print(f"Test failed with assertion error")
    except Exception as e:
        print(f"Test failed: {e}")

    print("\nTesting compress_cols preserves 2D structure...")
    try:
        test_compress_cols_preserves_2d_structure()
        print("Test passed!")
    except AssertionError as e:
        print(f"Test failed with assertion error")
    except Exception as e:
        print(f"Test failed: {e}")

    print("\nTesting compress_rowcols preserves 2D structure...")
    try:
        test_compress_rowcols_preserves_2d_structure()
        print("Test passed!")
    except AssertionError as e:
        print(f"Test failed with assertion error")
    except Exception as e:
        print(f"Test failed: {e}")
```

<details>

<summary>
**Failing input**: `masked_array(data=[[--]], mask=[[True]])`
</summary>
```
Testing compress_rows preserves 2D structure...
Falsifying example: test_compress_rows_preserves_2d_structure(
    arr=masked_array(data=[[--]],
                 mask=[[ True]],
           fill_value=1e+20,
                dtype=float64),
)
AssertionError: Expected 2D output, got ndim=1 for input [[--]]

Testing compress_cols preserves 2D structure...
Falsifying example: test_compress_cols_preserves_2d_structure(
    arr=masked_array(data=[[--]],
                 mask=[[ True]],
           fill_value=1e+20,
                dtype=float64),
)
AssertionError: Expected 2D output, got ndim=1 for input [[--]]

Testing compress_rowcols preserves 2D structure...
Falsifying example: test_compress_rowcols_preserves_2d_structure(
    arr=masked_array(data=[[--]],
                 mask=[[ True]],
           fill_value=1e+20,
                dtype=float64),
)
AssertionError: Expected 2D output, got ndim=1 for input [[--]]
```
</details>

## Reproducing the Bug

```python
import numpy as np
import numpy.ma as ma

# Test Case 1: Single masked element (1x1 array)
print("Test Case 1: Single masked element (1x1 array)")
arr = ma.array([[999]], mask=[[True]])
print(f"Input: {arr}")
print(f"Input shape: {arr.shape}, ndim={arr.ndim}")

result_rows = ma.compress_rows(arr)
result_cols = ma.compress_cols(arr)
result_rowcols = ma.compress_rowcols(arr)

print(f"compress_rows result: shape={result_rows.shape}, ndim={result_rows.ndim}, data={result_rows}")
print(f"compress_cols result: shape={result_cols.shape}, ndim={result_cols.ndim}, data={result_cols}")
print(f"compress_rowcols result: shape={result_rowcols.shape}, ndim={result_rowcols.ndim}, data={result_rowcols}")
print()

# Test Case 2: Fully masked 2x2 array
print("Test Case 2: Fully masked 2x2 array")
arr2 = ma.array([[1, 2], [3, 4]], mask=[[True, True], [True, True]])
print(f"Input: {arr2}")
print(f"Input shape: {arr2.shape}, ndim={arr2.ndim}")

result2_rows = ma.compress_rows(arr2)
result2_cols = ma.compress_cols(arr2)
result2_rowcols = ma.compress_rowcols(arr2)

print(f"compress_rows result: shape={result2_rows.shape}, ndim={result2_rows.ndim}, data={result2_rows}")
print(f"compress_cols result: shape={result2_cols.shape}, ndim={result2_cols.ndim}, data={result2_cols}")
print(f"compress_rowcols result: shape={result2_rowcols.shape}, ndim={result2_rowcols.ndim}, data={result2_rowcols}")
print()

# Test Case 3: Partially masked array (control case)
print("Test Case 3: Partially masked array (control case - should return 2D)")
arr3 = ma.array([[1, 2], [3, 4]], mask=[[True, False], [False, False]])
print(f"Input: {arr3}")
print(f"Input shape: {arr3.shape}, ndim={arr3.ndim}")

result3_rows = ma.compress_rows(arr3)
result3_cols = ma.compress_cols(arr3)
result3_rowcols = ma.compress_rowcols(arr3)

print(f"compress_rows result: shape={result3_rows.shape}, ndim={result3_rows.ndim}, data={result3_rows}")
print(f"compress_cols result: shape={result3_cols.shape}, ndim={result3_cols.ndim}, data={result3_cols}")
print(f"compress_rowcols result: shape={result3_rowcols.shape}, ndim={result3_rowcols.ndim}, data={result3_rowcols}")
print()

# Demonstrate the downstream error
print("Demonstrating downstream error when expecting 2D output:")
try:
    arr_fail = ma.array([[999]], mask=[[True]])
    result = ma.compress_rows(arr_fail)
    print(f"Result shape: {result.shape}")
    num_cols = result.shape[1]  # This will fail!
    print(f"Number of columns: {num_cols}")
except IndexError as e:
    print(f"ERROR: IndexError occurred - {e}")
    print("This happens because result.shape is (0,) which only has one dimension")
```

<details>

<summary>
Output showing dimension loss and resulting IndexError
</summary>
```
Test Case 1: Single masked element (1x1 array)
Input: [[--]]
Input shape: (1, 1), ndim=2
compress_rows result: shape=(0,), ndim=1, data=[]
compress_cols result: shape=(0,), ndim=1, data=[]
compress_rowcols result: shape=(0,), ndim=1, data=[]

Test Case 2: Fully masked 2x2 array
Input: [[-- --]
 [-- --]]
Input shape: (2, 2), ndim=2
compress_rows result: shape=(0,), ndim=1, data=[]
compress_cols result: shape=(0,), ndim=1, data=[]
compress_rowcols result: shape=(0,), ndim=1, data=[]

Test Case 3: Partially masked array (control case - should return 2D)
Input: [[-- 2]
 [3 4]]
Input shape: (2, 2), ndim=2
compress_rows result: shape=(1, 2), ndim=2, data=[[3 4]]
compress_cols result: shape=(2, 1), ndim=2, data=[[2]
 [4]]
compress_rowcols result: shape=(1, 1), ndim=2, data=[[4]]

Demonstrating downstream error when expecting 2D output:
Result shape: (0,)
ERROR: IndexError occurred - tuple index out of range
This happens because result.shape is (0,) which only has one dimension
```
</details>

## Why This Is A Bug

These functions violate the principle of type consistency by returning different dimensional arrays based on the data content rather than the operation performed. The functions are documented to:
1. Require 2D input arrays - they raise NotImplementedError for non-2D inputs
2. Perform row/column compression operations on 2D arrays
3. Return "compressed arrays"

The inconsistent behavior where the functions return 2D arrays when some data remains but 1D arrays when all data is masked violates reasonable expectations and causes real crashes in downstream code. Standard numpy array operations preserve dimensionality even for empty results (e.g., `arr[empty_index]` on a 2D array returns a 2D array with 0 rows).

This breaks common patterns like:
- Accessing `result.shape[1]` to get the number of columns after row compression
- Concatenating results from multiple compress operations
- Any code that assumes dimension preservation

## Relevant Context

The bug stems from line 948 in `/numpy/ma/extras.py` in the `compress_nd` function, which all three compress functions ultimately call:

```python
# All is masked: return empty
if m.all():
    return nxarray([])  # This returns a 1D array regardless of input dimensions
```

The functions work correctly when data remains because they use boolean indexing which preserves dimensions. However, the edge case handling for fully masked arrays incorrectly returns a 1D empty array.

NumPy documentation: https://numpy.org/doc/stable/reference/generated/numpy.ma.compress_rows.html
Source code: https://github.com/numpy/numpy/blob/main/numpy/ma/extras.py#L948

## Proposed Fix

```diff
--- a/numpy/ma/extras.py
+++ b/numpy/ma/extras.py
@@ -945,8 +945,20 @@ def compress_nd(x, axis=None):
         return x._data
     # All is masked: return empty
     if m.all():
-        return nxarray([])
+        # Preserve dimensionality when returning empty arrays
+        if axis is None:
+            # compress_rowcols case - both dimensions compressed
+            return np.empty((0, 0), dtype=x.dtype)
+        elif axis == 0:
+            # compress_rows case - preserve columns
+            return np.empty((0, x.shape[1]), dtype=x.dtype)
+        elif axis == 1:
+            # compress_cols case - preserve rows
+            return np.empty((x.shape[0], 0), dtype=x.dtype)
+        else:
+            # Handle other axis values appropriately
+            shape = list(x.shape)
+            shape[axis] = 0
+            return np.empty(shape, dtype=x.dtype)
     # Filter elements through boolean indexing
     data = x._data
```