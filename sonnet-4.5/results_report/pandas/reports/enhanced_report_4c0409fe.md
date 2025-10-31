# Bug Report: pandas.core.sparse.SparseArray.cumsum() Infinite Recursion Crash

**Target**: `pandas.core.sparse.api.SparseArray.cumsum`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `cumsum()` method on SparseArray crashes with infinite recursion (RecursionError) when called on arrays with non-null fill values, which includes the default configuration for integer arrays (fill_value=0).

## Property-Based Test

```python
import numpy as np
from pandas.core.sparse.api import SparseArray
from hypothesis import given, strategies as st

@given(st.lists(st.integers(min_value=-100, max_value=100), min_size=1, max_size=100))
def test_cumsum_preserves_length(data):
    sparse = SparseArray(data)
    result = sparse.cumsum()
    assert len(result) == len(sparse), \
        f"cumsum() changed length: {len(result)} != {len(sparse)}"

if __name__ == "__main__":
    test_cumsum_preserves_length()
```

<details>

<summary>
**Failing input**: `[0]` (or any integer list)
</summary>
```
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/40/hypo.py", line 13, in <module>
  |     test_cumsum_preserves_length()
  |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  |   File "/home/npc/pbt/agentic-pbt/worker_/40/hypo.py", line 6, in test_cumsum_preserves_length
  |     def test_cumsum_preserves_length(data):
  |                    ^^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
  |     raise the_error_hypothesis_found
  | hypothesis.errors.FlakyFailure: Inconsistent results from replaying a test case!
  |   last: INTERESTING from RecursionError at /home/npc/miniconda/lib/python3.13/site-packages/pandas/core/dtypes/cast.py:1810
  |   this: INTERESTING from RecursionError at /home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/getlimits.py:699 (3 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1207, in _execute_once_for_engine
    |     result = self.execute_once(data)
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1147, in execute_once
    |     result = self.test_runner(data, run)
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 822, in default_executor
    |     return function(data)
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1103, in run
    |     return test(*args, **kwargs)
    |   File "/home/npc/pbt/agentic-pbt/worker_/40/hypo.py", line 6, in test_cumsum_preserves_length
    |     def test_cumsum_preserves_length(data):
    |                    ^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1012, in test
    |     result = self.test(*args, **kwargs)
    |   File "/home/npc/pbt/agentic-pbt/worker_/40/hypo.py", line 8, in test_cumsum_preserves_length
    |     result = sparse.cumsum()
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/sparse/array.py", line 1550, in cumsum
    |     return SparseArray(self.to_dense()).cumsum()
    |            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/sparse/array.py", line 1550, in cumsum
    |     return SparseArray(self.to_dense()).cumsum()
    |            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/sparse/array.py", line 1550, in cumsum
    |     return SparseArray(self.to_dense()).cumsum()
    |            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
    |   [Previous line repeated 1990 more times]
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/sparse/array.py", line 495, in __init__
    |     self._dtype = SparseDtype(sparse_values.dtype, fill_value)
    |                   ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/dtypes/dtypes.py", line 1689, in __init__
    |     self._check_fill_value()
    |     ~~~~~~~~~~~~~~~~~~~~~~^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/dtypes/dtypes.py", line 1777, in _check_fill_value
    |     if not can_hold_element(dummy, val):
    |            ~~~~~~~~~~~~~~~~^^^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/dtypes/cast.py", line 1772, in can_hold_element
    |     np_can_hold_element(dtype, element)
    |     ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/dtypes/cast.py", line 1810, in np_can_hold_element
    |     info = np.iinfo(dtype)
    | RecursionError: maximum recursion depth exceeded
    +---------------- 2 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1207, in _execute_once_for_engine
    |     result = self.execute_once(data)
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1147, in execute_once
    |     result = self.test_runner(data, run)
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 822, in default_executor
    |     return function(data)
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1103, in run
    |     return test(*args, **kwargs)
    |   File "/home/npc/pbt/agentic-pbt/worker_/40/hypo.py", line 6, in test_cumsum_preserves_length
    |     def test_cumsum_preserves_length(data):
    |                    ^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1012, in test
    |     result = self.test(*args, **kwargs)
    |   File "/home/npc/pbt/agentic-pbt/worker_/40/hypo.py", line 8, in test_cumsum_preserves_length
    |     result = sparse.cumsum()
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/sparse/array.py", line 1550, in cumsum
    |     return SparseArray(self.to_dense()).cumsum()
    |            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/sparse/array.py", line 1550, in cumsum
    |     return SparseArray(self.to_dense()).cumsum()
    |            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/sparse/array.py", line 1550, in cumsum
    |     return SparseArray(self.to_dense()).cumsum()
    |            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
    |   [Previous line repeated 1989 more times]
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/sparse/array.py", line 495, in __init__
    |     self._dtype = SparseDtype(sparse_values.dtype, fill_value)
    |                   ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/dtypes/dtypes.py", line 1689, in __init__
    |     self._check_fill_value()
    |     ~~~~~~~~~~~~~~~~~~~~~~^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/dtypes/dtypes.py", line 1777, in _check_fill_value
    |     if not can_hold_element(dummy, val):
    |            ~~~~~~~~~~~~~~~~^^^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/dtypes/cast.py", line 1772, in can_hold_element
    |     np_can_hold_element(dtype, element)
    |     ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/dtypes/cast.py", line 1810, in np_can_hold_element
    |     info = np.iinfo(dtype)
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/getlimits.py", line 699, in __init__
    |     try:
    |     ...<2 lines>...
    |         self.dtype = numeric.dtype(type(int_type))
    | RecursionError: maximum recursion depth exceeded
    +---------------- 3 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1207, in _execute_once_for_engine
    |     result = self.execute_once(data)
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1147, in execute_once
    |     result = self.test_runner(data, run)
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 822, in default_executor
    |     return function(data)
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1103, in run
    |     return test(*args, **kwargs)
    |   File "/home/npc/pbt/agentic-pbt/worker_/40/hypo.py", line 6, in test_cumsum_preserves_length
    |     def test_cumsum_preserves_length(data):
    |                    ^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1012, in test
    |     result = self.test(*args, **kwargs)
    |   File "/home/npc/pbt/agentic-pbt/worker_/40/hypo.py", line 8, in test_cumsum_preserves_length
    |     result = sparse.cumsum()
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/sparse/array.py", line 1550, in cumsum
    |     return SparseArray(self.to_dense()).cumsum()
    |            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/sparse/array.py", line 1550, in cumsum
    |     return SparseArray(self.to_dense()).cumsum()
    |            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/sparse/array.py", line 1550, in cumsum
    |     return SparseArray(self.to_dense()).cumsum()
    |            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
    |   [Previous line repeated 1989 more times]
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/sparse/array.py", line 495, in __init__
    |     self._dtype = SparseDtype(sparse_values.dtype, fill_value)
    |                   ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/dtypes/dtypes.py", line 1689, in __init__
    |     self._check_fill_value()
    |     ~~~~~~~~~~~~~~~~~~~~~~^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/dtypes/dtypes.py", line 1777, in _check_fill_value
    |     if not can_hold_element(dummy, val):
    |            ~~~~~~~~~~~~~~~~^^^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/dtypes/cast.py", line 1772, in can_hold_element
    |     np_can_hold_element(dtype, element)
    |     ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/dtypes/cast.py", line 1810, in np_can_hold_element
    |     info = np.iinfo(dtype)
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/getlimits.py", line 699, in __init__
    |     try:
    |     ...<2 lines>...
    |         self.dtype = numeric.dtype(type(int_type))
    | RecursionError: maximum recursion depth exceeded
    +------------------------------------
```
</details>

## Reproducing the Bug

```python
from pandas.core.sparse.api import SparseArray

sparse = SparseArray([1, 2, 3])
result = sparse.cumsum()
```

<details>

<summary>
RecursionError: maximum recursion depth exceeded
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/40/repo.py", line 4, in <module>
    result = sparse.cumsum()
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/sparse/array.py", line 1550, in cumsum
    return SparseArray(self.to_dense()).cumsum()
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/sparse/array.py", line 1550, in cumsum
    return SparseArray(self.to_dense()).cumsum()
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/sparse/array.py", line 1550, in cumsum
    return SparseArray(self.to_dense()).cumsum()
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  [Previous line repeated 989 more times]
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/sparse/array.py", line 495, in __init__
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

## Why This Is A Bug

The `cumsum()` method is a documented public API method that should compute the cumulative sum of array elements. However, it crashes with infinite recursion for the most common use case - integer sparse arrays with their default fill_value of 0.

The bug occurs at line 1550 in `/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/sparse/array.py`:

```python
if not self._null_fill_value:
    return SparseArray(self.to_dense()).cumsum()
```

When `_null_fill_value` is False (which happens when fill_value is 0 for integer arrays), the code attempts to:
1. Convert the sparse array to dense with `self.to_dense()`
2. Create a new SparseArray from the dense array
3. Call `cumsum()` on the new SparseArray

The problem is that the new SparseArray created from the dense array still has `_null_fill_value = False` (because it infers fill_value=0 for integer arrays), causing it to recursively hit the same code path infinitely.

This violates the expected behavior documented in pandas where `cumsum()` should compute cumulative sums for all valid numeric arrays, sparse or dense.

## Relevant Context

- SparseArrays with integer dtype default to `fill_value=0`, which is considered non-null
- The `_null_fill_value` property returns `False` for non-null fill values like 0
- This affects all integer SparseArrays unless explicitly created with `fill_value=np.nan`
- The bug only affects arrays with non-null fill values; arrays with null fill values (like NaN) work correctly
- This is a regression or oversight - the cumsum operation has well-defined mathematical semantics for all numeric values

Documentation reference: https://pandas.pydata.org/docs/reference/api/pandas.arrays.SparseArray.cumsum.html

## Proposed Fix

```diff
--- a/pandas/core/arrays/sparse/array.py
+++ b/pandas/core/arrays/sparse/array.py
@@ -1547,7 +1547,7 @@ class SparseArray(OpsMixin, PandasObject, ExtensionArray):
         if axis is not None and axis >= self.ndim:  # Mimic ndarray behaviour.
             raise ValueError(f"axis(={axis}) out of bounds")

         if not self._null_fill_value:
-            return SparseArray(self.to_dense()).cumsum()
+            return SparseArray(self.to_dense().cumsum())

         return SparseArray(
             self.sp_values.cumsum(),
```

The fix calls `cumsum()` on the dense numpy array first to compute the cumulative sum, then wraps the result in a SparseArray. This avoids the infinite recursion by ensuring cumsum() is never called recursively on another SparseArray with a non-null fill value.