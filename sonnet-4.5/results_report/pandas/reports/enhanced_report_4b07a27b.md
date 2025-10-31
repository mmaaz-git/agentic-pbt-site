# Bug Report: pandas.core.arrays.sparse.array.SparseArray.cumsum Infinite Recursion

**Target**: `pandas.core.arrays.sparse.array.SparseArray.cumsum`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `cumsum()` method on `SparseArray` crashes with infinite recursion when the sparse array has a non-null fill value (such as the default fill_value=0 for integer arrays), making cumulative sum completely unusable for integer sparse arrays.

## Property-Based Test

```python
from pandas.arrays import SparseArray
from hypothesis import given, strategies as st, settings

@given(
    st.lists(st.integers(min_value=0, max_value=100), min_size=1, max_size=50)
)
@settings(max_examples=500)
def test_cumsum_length(data):
    arr = SparseArray(data)
    cumsum_result = arr.cumsum()
    assert len(cumsum_result) == len(arr)

if __name__ == "__main__":
    test_cumsum_length()
```

<details>

<summary>
**Failing input**: `[0]`
</summary>
```
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/56/hypo.py", line 14, in <module>
  |     test_cumsum_length()
  |     ~~~~~~~~~~~~~~~~~~^^
  |   File "/home/npc/pbt/agentic-pbt/worker_/56/hypo.py", line 5, in test_cumsum_length
  |     st.lists(st.integers(min_value=0, max_value=100), min_size=1, max_size=50)
  |                ^^^
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
    |   File "/home/npc/pbt/agentic-pbt/worker_/56/hypo.py", line 5, in test_cumsum_length
    |     st.lists(st.integers(min_value=0, max_value=100), min_size=1, max_size=50)
    |                ^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1012, in test
    |     result = self.test(*args, **kwargs)
    |   File "/home/npc/pbt/agentic-pbt/worker_/56/hypo.py", line 10, in test_cumsum_length
    |     cumsum_result = arr.cumsum()
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
    |   File "/home/npc/pbt/agentic-pbt/worker_/56/hypo.py", line 5, in test_cumsum_length
    |     st.lists(st.integers(min_value=0, max_value=100), min_size=1, max_size=50)
    |                ^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1012, in test
    |     result = self.test(*args, **kwargs)
    |   File "/home/npc/pbt/agentic-pbt/worker_/56/hypo.py", line 10, in test_cumsum_length
    |     cumsum_result = arr.cumsum()
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
    |   File "/home/npc/pbt/agentic-pbt/worker_/56/hypo.py", line 5, in test_cumsum_length
    |     st.lists(st.integers(min_value=0, max_value=100), min_size=1, max_size=50)
    |                ^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1012, in test
    |     result = self.test(*args, **kwargs)
    |   File "/home/npc/pbt/agentic-pbt/worker_/56/hypo.py", line 10, in test_cumsum_length
    |     cumsum_result = arr.cumsum()
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
from pandas.arrays import SparseArray

arr = SparseArray([1, 2, 3], fill_value=0)
result = arr.cumsum()
print(result)
```

<details>

<summary>
RecursionError: maximum recursion depth exceeded
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/56/repo.py", line 4, in <module>
    result = arr.cumsum()
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

The `cumsum()` method is expected to compute the cumulative sum of array elements, as documented in pandas and consistent with numpy's `cumsum()` behavior. For example, `SparseArray([1, 2, 3]).cumsum()` should return a SparseArray equivalent to `[1, 3, 6]`. Instead, the method immediately crashes with infinite recursion.

The bug specifically affects integer sparse arrays because they have a default `fill_value=0`, which is non-null. When `self._null_fill_value` is `False` (line 1549), the code attempts to convert to a dense array and then recursively call `cumsum()`:

```python
if not self._null_fill_value:
    return SparseArray(self.to_dense()).cumsum()  # Line 1550
```

The problem is that `SparseArray(self.to_dense())` creates a NEW SparseArray from the dense numpy array, which automatically infers the same `fill_value=0` for integer arrays. This new SparseArray also has `_null_fill_value=False`, triggering the same code path and causing infinite recursion.

In contrast, float arrays work correctly because their default `fill_value=NaN` is considered null, so they take the alternative code path (lines 1552-1556) that properly computes the cumulative sum.

## Relevant Context

- **Affected arrays**: All integer SparseArrays with non-null fill values (default behavior)
- **Unaffected arrays**: Float SparseArrays with NaN fill value work correctly
- **Documentation reference**: SparseArray is documented at [pandas.arrays.SparseArray](https://pandas.pydata.org/docs/reference/api/pandas.arrays.SparseArray.html)
- **Source code location**: `/pandas/core/arrays/sparse/array.py:1550`
- **Default fill values**: Integer arrays default to `fill_value=0`, float arrays default to `fill_value=NaN`

The `_null_fill_value` property (line 683) returns `True` only when the fill value is considered "null" (NaN for floats, None for objects). For integer arrays with `fill_value=0`, this returns `False`.

## Proposed Fix

```diff
--- a/pandas/core/arrays/sparse/array.py
+++ b/pandas/core/arrays/sparse/array.py
@@ -1547,7 +1547,8 @@ class SparseArray(OpsMixin, PandasObject, ExtensionArray):
             raise ValueError(f"axis(={axis}) out of bounds")

         if not self._null_fill_value:
-            return SparseArray(self.to_dense()).cumsum()
+            import numpy as np
+            return SparseArray(np.cumsum(self.to_dense()))

         return SparseArray(
             self.sp_values.cumsum(),
```