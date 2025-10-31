# Bug Report: numpy.ma.compress_rows and compress_cols Return Wrong Dimensionality for Fully Masked Arrays

**Target**: `numpy.ma.compress_rows`, `numpy.ma.compress_cols`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`compress_rows` and `compress_cols` return 1-D arrays with shape `(0,)` when operating on fully-masked 2-D arrays, instead of maintaining the expected 2-D structure with shapes `(0, cols)` or `(rows, 0)` respectively.

## Property-Based Test

```python
import numpy as np
import numpy.ma as ma
from hypothesis import given, strategies as st

@given(st.integers(min_value=2, max_value=10), st.integers(min_value=2, max_value=10))
def test_compress_rows_shape_inconsistency_when_all_masked(rows, cols):
    data = np.zeros((rows, cols))
    mask = np.ones((rows, cols), dtype=bool)

    arr = ma.array(data, mask=mask)
    result = ma.compress_rows(arr)

    assert result.ndim == 2, f"Expected 2-D array but got {result.ndim}-D array"
    assert result.shape == (0, cols), f"Expected shape (0, {cols}) but got {result.shape}"

@given(st.integers(min_value=2, max_value=10), st.integers(min_value=2, max_value=10))
def test_compress_cols_shape_inconsistency_when_all_masked(rows, cols):
    data = np.zeros((rows, cols))
    mask = np.ones((rows, cols), dtype=bool)

    arr = ma.array(data, mask=mask)
    result = ma.compress_cols(arr)

    assert result.ndim == 2, f"Expected 2-D array but got {result.ndim}-D array"
    assert result.shape == (rows, 0), f"Expected shape ({rows}, 0) but got {result.shape}"

if __name__ == "__main__":
    print("Testing compress_rows shape consistency...")
    test_compress_rows_shape_inconsistency_when_all_masked()

    print("Testing compress_cols shape consistency...")
    test_compress_cols_shape_inconsistency_when_all_masked()
```

<details>

<summary>
**Failing input**: `rows=2, cols=2` (or any dimensions with fully masked array)
</summary>
```
Testing compress_rows shape consistency...
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/39/hypo.py", line 29, in <module>
    test_compress_rows_shape_inconsistency_when_all_masked()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/39/hypo.py", line 6, in test_compress_rows_shape_inconsistency_when_all_masked
    def test_compress_rows_shape_inconsistency_when_all_masked(rows, cols):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/39/hypo.py", line 13, in test_compress_rows_shape_inconsistency_when_all_masked
    assert result.ndim == 2, f"Expected 2-D array but got {result.ndim}-D array"
           ^^^^^^^^^^^^^^^^
AssertionError: Expected 2-D array but got 1-D array
Falsifying example: test_compress_rows_shape_inconsistency_when_all_masked(
    # The test always failed when commented parts were varied together.
    rows=2,  # or any other generated value
    cols=2,  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
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
```

<details>

<summary>
Output shows 1-D array returned for fully masked input, causing IndexError
</summary>
```
Testing numpy.ma.compress_rows and compress_cols shape inconsistency
======================================================================

Test 1: Fully masked 2x3 array
----------------------------------------
Input shape: (2, 3)
Input array:
[[-- -- --]
 [-- -- --]]
Input mask:
[[ True  True  True]
 [ True  True  True]]

compress_rows result shape: (0,)
compress_rows result: []
Expected shape: (0, 3)
Result dimensionality: 1

compress_cols result shape: (0,)
compress_cols result: []
Expected shape: (2, 0)
Result dimensionality: 1


Test 2: Partially masked 2x3 array (for comparison)
----------------------------------------
Input shape: (2, 3)
Input array:
[[-- 2.0 3.0]
 [4.0 -- 6.0]]
Input mask:
[[ True False False]
 [False  True False]]

compress_rows result shape: (0, 3)
compress_rows result:
[]
Result dimensionality: 2

compress_cols result shape: (2, 1)
compress_cols result:
[[3.]
 [6.]]
Result dimensionality: 2


Test 3: Demonstrating downstream failure
----------------------------------------
Attempting to access shape[1] on fully masked result:
Result shape: (0,)
ERROR: IndexError occurred - tuple index out of range
This would break code expecting a 2-D array!
```
</details>

## Why This Is A Bug

This violates the expected behavior of `compress_rows` and `compress_cols` in multiple ways:

1. **Dimensionality Contract Violation**: These functions are explicitly documented to work with 2-D arrays and should maintain 2-D output structure. The functions raise `NotImplementedError` for non-2D inputs, establishing a clear 2-D contract.

2. **Inconsistent Behavior**: The functions return 2-D arrays when partially masked but 1-D arrays when fully masked. This inconsistency is not documented and breaks the principle of least surprise.

3. **Semantic Violation**: Functions named `compress_rows` and `compress_cols` inherently operate on rows and columns - 2-D concepts. A 1-D array has neither rows nor columns.

4. **Documentation Contradiction**: All examples in the documentation show 2-D outputs. The docstrings state these functions "suppress whole rows/columns" which implies maintaining the 2-D structure with fewer rows or columns.

5. **Downstream Failures**: Code that correctly expects `shape[1]` to exist (since input must be 2-D) will crash with `IndexError: tuple index out of range`.

## Relevant Context

The root cause is in the `compress_nd` function at line 948 of `/home/npc/pbt/agentic-pbt/envs/numpy_env/lib/python3.13/site-packages/numpy/ma/extras.py`:

```python
# All is masked: return empty
if m.all():
    return nxarray([])  # This always returns 1-D array
```

This unconditionally returns a 1-D empty array, ignoring the `axis` parameter and original array dimensions.

The functions work correctly for partially masked arrays because the filtering logic preserves dimensions:
```python
for ax in axis:
    axes = tuple(list(range(ax)) + list(range(ax + 1, x.ndim)))
    data = data[(slice(None),) * ax + (~m.any(axis=axes),)]
```

Documentation references:
- numpy.ma.compress_rows: https://numpy.org/doc/stable/reference/generated/numpy.ma.compress_rows.html
- numpy.ma.compress_cols: https://numpy.org/doc/stable/reference/generated/numpy.ma.compress_cols.html

## Proposed Fix

```diff
--- a/numpy/ma/extras.py
+++ b/numpy/ma/extras.py
@@ -944,8 +944,18 @@ def compress_nd(x, axis=None):
     # Nothing is masked: return x
     if m is nomask or not m.any():
         return x._data
     # All is masked: return empty
     if m.all():
-        return nxarray([])
+        # Preserve the shape based on the axis parameter
+        if axis is None:
+            # All axes compressed: return empty array with all dims 0
+            new_shape = tuple(0 for _ in x.shape)
+        else:
+            # Only specified axes compressed: preserve other dimensions
+            new_shape = list(x.shape)
+            for ax in axis:
+                new_shape[ax] = 0
+            new_shape = tuple(new_shape)
+        return np.empty(new_shape, dtype=x.dtype)
     # Filter elements through boolean indexing
     data = x._data
     for ax in axis:
```