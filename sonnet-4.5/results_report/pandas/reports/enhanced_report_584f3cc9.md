# Bug Report: pandas.core.arrays.sparse.SparseArray.cumsum Infinite Recursion with Non-Null Fill Values

**Target**: `pandas.core.arrays.sparse.SparseArray.cumsum`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

SparseArray.cumsum() triggers infinite recursion for any array with non-null fill values (e.g., fill_value=0), resulting in RecursionError and complete failure of the cumulative sum operation.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pandas.core.arrays.sparse import SparseArray
import numpy as np

@given(st.lists(st.integers(min_value=-1000, max_value=1000), min_size=1, max_size=100))
def test_cumsum_matches_dense(values):
    """Test that SparseArray.cumsum() matches the dense array cumsum."""
    sparse_arr = SparseArray(values, fill_value=0)

    # Calculate cumsum on sparse array
    try:
        sparse_cumsum = sparse_arr.cumsum().to_dense()
    except RecursionError:
        print(f"\nRecursionError on input: {values[:5]}{'...' if len(values) > 5 else ''}")
        raise

    # Calculate cumsum on dense array
    dense_cumsum = sparse_arr.to_dense().cumsum()

    # They should be equal
    assert np.array_equal(sparse_cumsum, dense_cumsum), \
        f"cumsum() on sparse should match cumsum() on dense\nSparse: {sparse_cumsum}\nDense: {dense_cumsum}"
```

<details>

<summary>
**Failing input**: `[0]` (or any integer list including `[1, 2, 3]`)
</summary>
```
Running hypothesis test with minimal failing example...
Testing with values: [1, 2, 3]

RecursionError on input: [0]

RecursionError on input: [1000, 414, 891]

RecursionError on input: [414, 414, 891]

RecursionError on input: [414, 414, 414]

RecursionError on input: [-124, 395, -965]

RecursionError on input: [-124, -965, -965]

RecursionError on input: [-958, 629, 137, -117]

RecursionError on input: [0]

Hypothesis test failed with: Inconsistent results from replaying a test case!
  last: INTERESTING from RecursionError at /home/npc/miniconda/lib/python3.13/site-packages/pandas/core/dtypes/cast.py:1810
  this: INTERESTING from RecursionError at /home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/getlimits.py:699 (3 sub-exceptions)

Trying direct test with [1, 2, 3]...
RecursionError on input: [1, 2, 3]

Traceback (last 5 frames):
    self._dtype = SparseDtype(sparse_values.dtype, fill_value)
                  ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/dtypes/dtypes.py", line 1689, in __init__
    self._check_fill_value()
    ~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/dtypes/dtypes.py", line 1777, in _check_fill_value
    if not can_hold_element(dummy, val):
           ~~~~~~~~~~~~~~~~^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/dtypes/cast.py", line 1772, in can_hold_element
    np_can_hold_element(dtype, element)
    ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/dtypes/cast.py", line 1810, in np_can_hold_element
    info = np.iinfo(dtype)
RecursionError: maximum recursion depth exceeded
```
</details>

## Reproducing the Bug

```python
from pandas.core.arrays.sparse import SparseArray

# Create a SparseArray with non-null fill value (0)
sparse_arr = SparseArray([1, 2, 3], fill_value=0)
print(f"Sparse array: {sparse_arr.to_dense()}")
print(f"Fill value: {sparse_arr.fill_value}")
print(f"_null_fill_value: {sparse_arr._null_fill_value}")

# Attempt to calculate cumulative sum
try:
    result = sparse_arr.cumsum()
    print(f"Success: {result.to_dense()}")
except RecursionError as e:
    print(f"RecursionError: maximum recursion depth exceeded")
    print(f"Error type: {type(e).__name__}")
```

<details>

<summary>
RecursionError: maximum recursion depth exceeded
</summary>
```
Sparse array: [1 2 3]
Fill value: 0
_null_fill_value: False
RecursionError: maximum recursion depth exceeded
Error type: RecursionError
```
</details>

## Why This Is A Bug

The bug violates the expected behavior of cumulative sum operations on sparse arrays. The documentation states that `cumsum()` should return a SparseArray containing the cumulative sums, but instead it crashes with infinite recursion.

The root cause is in line 1550 of `/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/sparse/array.py`:

```python
if not self._null_fill_value:
    return SparseArray(self.to_dense()).cumsum()
```

This code path is triggered when the fill value is non-null (e.g., 0 instead of NaN). The issue is that:

1. `SparseArray([1,2,3], fill_value=0)` has `_null_fill_value == False` because 0 is not NaN
2. The method calls `SparseArray(self.to_dense())` which creates a new SparseArray from the dense array `[1,2,3]`
3. This new SparseArray **inherits the same fill_value=0** from the original array
4. Therefore, the new SparseArray also has `_null_fill_value == False`
5. When `.cumsum()` is called on this new SparseArray, it enters the same conditional branch again
6. This creates another SparseArray and calls `.cumsum()` on it, leading to infinite recursion

The intended behavior was clearly to compute the cumulative sum on the dense array and then convert back to sparse, but the implementation incorrectly calls `.cumsum()` on a new SparseArray instead of on the dense array itself.

## Relevant Context

Sparse arrays with fill_value=0 are fundamental to sparse matrix operations in scientific computing and data analysis. This is arguably the most common use case for sparse arrays, as sparse matrices typically represent zero values implicitly. The cumulative sum is a basic array operation that should work for all valid sparse arrays.

The `_null_fill_value` property returns `True` when the fill value is NaN/null (the default), and `False` for non-null values like 0. The code correctly identifies that special handling is needed for non-null fill values, but the implementation contains a critical error.

Documentation for SparseArray.cumsum: https://pandas.pydata.org/docs/reference/api/pandas.arrays.SparseArray.cumsum.html

Source code location: `/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/sparse/array.py:1540-1556`

## Proposed Fix

```diff
--- a/pandas/core/arrays/sparse/array.py
+++ b/pandas/core/arrays/sparse/array.py
@@ -1547,7 +1547,7 @@ class SparseArray(OpsMixin, PandasObject, ExtensionArray):
             raise ValueError(f"axis(={axis}) out of bounds")

         if not self._null_fill_value:
-            return SparseArray(self.to_dense()).cumsum()
+            return SparseArray(self.to_dense().cumsum())

         return SparseArray(
             self.sp_values.cumsum(),
```