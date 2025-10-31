# Bug Report: pandas.core.arrays.sparse.SparseArray.cumsum Infinite Recursion

**Target**: `pandas.core.arrays.sparse.SparseArray.cumsum`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`SparseArray.cumsum()` causes infinite recursion and crashes with RecursionError when called on sparse arrays with non-NA fill values (e.g., integers with default fill_value=0), making the method completely unusable for the most common sparse array types.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages')

import numpy as np
from hypothesis import given, strategies as st, settings, example
from pandas.arrays import SparseArray


@given(st.lists(st.integers(min_value=0, max_value=100), min_size=1, max_size=50))
@example([0])  # Minimal example that fails
@settings(max_examples=1)
def test_cumsum_matches_dense(data):
    arr = SparseArray(data)
    dense = arr.to_dense()

    sparse_cumsum = arr.cumsum()
    dense_cumsum = np.cumsum(dense)

    assert np.array_equal(sparse_cumsum.to_dense(), dense_cumsum)


if __name__ == "__main__":
    test_cumsum_matches_dense()
```

<details>

<summary>
**Failing input**: `data=[0]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/26/hypo.py", line 23, in <module>
    test_cumsum_matches_dense()
    ~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/26/hypo.py", line 10, in test_cumsum_matches_dense
    @example([0])  # Minimal example that fails
                   ^^^
  File "/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/hypothesis/core.py", line 2062, in wrapped_test
    _raise_to_user(errors, state.settings, [], " in explicit examples")
    ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/hypothesis/core.py", line 1613, in _raise_to_user
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/26/hypo.py", line 16, in test_cumsum_matches_dense
    sparse_cumsum = arr.cumsum()
  File "/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/core/arrays/sparse/array.py", line 1550, in cumsum
    return SparseArray(self.to_dense()).cumsum()
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/core/arrays/sparse/array.py", line 1550, in cumsum
    return SparseArray(self.to_dense()).cumsum()
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/core/arrays/sparse/array.py", line 1550, in cumsum
    return SparseArray(self.to_dense()).cumsum()
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  [Previous line repeated 1990 more times]
  File "/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/core/arrays/sparse/array.py", line 495, in __init__
    self._dtype = SparseDtype(sparse_values.dtype, fill_value)
                  ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/core/dtypes/dtypes.py", line 1689, in __init__
    self._check_fill_value()
    ~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/core/dtypes/dtypes.py", line 1777, in _check_fill_value
    if not can_hold_element(dummy, val):
           ~~~~~~~~~~~~~~~~^^^^^^^^^^^^
  File "/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/core/dtypes/cast.py", line 1772, in can_hold_element
    np_can_hold_element(dtype, element)
    ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^
  File "/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/core/dtypes/cast.py", line 1810, in np_can_hold_element
    info = np.iinfo(dtype)
RecursionError: maximum recursion depth exceeded
Falsifying explicit example: test_cumsum_matches_dense(
    data=[0],
)
```
</details>

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages')

from pandas.arrays import SparseArray
import traceback

# Create a SparseArray with integer data
# Default fill_value for integers is 0
arr = SparseArray([0, 1, 2], fill_value=0)
print(f"Created SparseArray: {arr}")
print(f"Fill value: {arr.fill_value}")
print(f"Sparse dtype: {arr.dtype}")
print(f"_null_fill_value: {arr._null_fill_value}")
print()

# Try to call cumsum - this should trigger infinite recursion
try:
    print("Calling cumsum()...")
    result = arr.cumsum()
    print(f"Result: {result}")
except RecursionError as e:
    print("RecursionError caught!")
    print(f"Error: {e}")
    print()
    print("Traceback (last 10 lines):")
    tb_lines = traceback.format_exc().split('\n')
    # Show relevant lines from the traceback
    for line in tb_lines[-15:]:
        if line:
            print(line)
```

<details>

<summary>
RecursionError: maximum recursion depth exceeded
</summary>
```
Created SparseArray: [0, 1, 2]
Fill: 0
IntIndex
Indices: array([1, 2], dtype=int32)

Fill value: 0
Sparse dtype: Sparse[int64, 0]
_null_fill_value: False

Calling cumsum()...
RecursionError caught!
Error: maximum recursion depth exceeded

Traceback (last 10 lines):
    self._dtype = SparseDtype(sparse_values.dtype, fill_value)
                  ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/core/dtypes/dtypes.py", line 1689, in __init__
    self._check_fill_value()
    ~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/core/dtypes/dtypes.py", line 1777, in _check_fill_value
    if not can_hold_element(dummy, val):
           ~~~~~~~~~~~~~~~~^^^^^^^^^^^^
  File "/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/core/dtypes/cast.py", line 1772, in can_hold_element
    np_can_hold_element(dtype, element)
    ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^
  File "/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/core/dtypes/cast.py", line 1810, in np_can_hold_element
    info = np.iinfo(dtype)
RecursionError: maximum recursion depth exceeded
```
</details>

## Why This Is A Bug

The bug occurs in the `cumsum` method at lines 1549-1550 of `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/core/arrays/sparse/array.py`:

```python
if not self._null_fill_value:
    return SparseArray(self.to_dense()).cumsum()
```

When a SparseArray has a non-NA fill value (which is determined by `_null_fill_value` being False), the code attempts to:
1. Convert the sparse array to dense using `self.to_dense()`
2. Create a new SparseArray from the dense array
3. Call cumsum() on the new SparseArray

The critical issue is that when creating a new SparseArray from a dense numpy array without explicitly specifying a fill_value, pandas automatically determines the fill_value based on the dtype. For integer arrays, this defaults to 0 (as shown by `na_value_for_dtype(np.dtype('int64'))` returning 0).

This means the newly created SparseArray will also have `_null_fill_value=False` (since 0 is not NA), causing it to enter the same code path again, resulting in infinite recursion.

This violates the expected behavior of cumsum, which should compute the cumulative sum of the array elements. According to pandas documentation, SparseArray should support standard array operations like cumsum. The method is completely broken for any sparse array with integer, boolean, or other non-floating-point dtypes that use non-NA fill values by default.

## Relevant Context

- The default fill_value for different dtypes in pandas SparseArrays:
  - int64: 0
  - float64: nan
  - bool: False
- Only float arrays with NaN fill values bypass this bug because `_null_fill_value` is True for NaN
- The bug affects the most common use case for sparse arrays - representing sparse integer matrices
- Documentation: https://pandas.pydata.org/docs/reference/api/pandas.arrays.SparseArray.html
- Source code location: `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/core/arrays/sparse/array.py:1549-1550`

## Proposed Fix

```diff
--- a/pandas/core/arrays/sparse/array.py
+++ b/pandas/core/arrays/sparse/array.py
@@ -1547,7 +1547,9 @@ class SparseArray(OpsMixin, PandasObject, ExtensionArray):
             raise ValueError(f"axis(={axis}) out of bounds")

         if not self._null_fill_value:
-            return SparseArray(self.to_dense()).cumsum()
+            # Compute cumsum on dense array and convert back to sparse
+            # Use NaN as fill_value to avoid recursion
+            return type(self)(np.cumsum(self.to_dense()), fill_value=np.nan)

         return SparseArray(
             self.sp_values.cumsum(),
```