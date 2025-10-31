# Bug Report: pandas.core.arrays.sparse.SparseArray.cumsum Infinite Recursion with Non-Null Fill Values

**Target**: `pandas.core.arrays.sparse.SparseArray.cumsum`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `cumsum()` method on `SparseArray` causes infinite recursion and crashes with `RecursionError` when called on sparse arrays that have non-null fill values, which includes all integer sparse arrays by default.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import pandas.core.arrays.sparse as sparse
import numpy as np

@given(
    data=st.lists(st.integers(min_value=-100, max_value=100), min_size=1, max_size=50),
    fill_value=st.integers(min_value=-10, max_value=10).filter(lambda x: x not in [np.nan])
)
@settings(max_examples=10, deadline=1000)  # Limit examples to avoid too many recursion errors
def test_sparse_array_cumsum_should_not_crash(data, fill_value):
    """cumsum should work on sparse arrays with non-null fill values"""
    np_data = np.array(data)
    sparse_arr = sparse.SparseArray(np_data, fill_value=fill_value)

    try:
        result = sparse_arr.cumsum()
        assert len(result) == len(sparse_arr)
        print(f"✓ Passed: data={data[:3]}{'...' if len(data) > 3 else ''}, fill_value={fill_value}")
    except RecursionError as e:
        print(f"✗ RecursionError: data={data[:3]}{'...' if len(data) > 3 else ''}, fill_value={fill_value}")
        raise
    except Exception as e:
        print(f"✗ {type(e).__name__}: data={data[:3]}{'...' if len(data) > 3 else ''}, fill_value={fill_value}")
        raise

if __name__ == "__main__":
    test_sparse_array_cumsum_should_not_crash()
```

<details>

<summary>
**Failing input**: `data=[0], fill_value=0`
</summary>
```
✗ RecursionError: data=[0], fill_value=0
✗ RecursionError: data=[20, -98, 41]..., fill_value=1
✗ RecursionError: data=[20, -98, 41]..., fill_value=1
✗ RecursionError: data=[20, -98, 41]..., fill_value=1
✗ RecursionError: data=[20, -98, 41]..., fill_value=1
✗ RecursionError: data=[20, -98, 91]..., fill_value=1
✗ RecursionError: data=[-100, -98, 91]..., fill_value=1
✗ RecursionError: data=[-100, -98, 91]..., fill_value=0
✗ RecursionError: data=[43, 35, -100], fill_value=6
✗ RecursionError: data=[0], fill_value=0
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/7/hypo.py", line 27, in <module>
  |     test_sparse_array_cumsum_should_not_crash()
  |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  |   File "/home/npc/pbt/agentic-pbt/worker_/7/hypo.py", line 6, in test_sparse_array_cumsum_should_not_crash
  |     data=st.lists(st.integers(min_value=-100, max_value=100), min_size=1, max_size=50),
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
    |   File "/home/npc/pbt/agentic-pbt/worker_/7/hypo.py", line 6, in test_sparse_array_cumsum_should_not_crash
    |     data=st.lists(st.integers(min_value=-100, max_value=100), min_size=1, max_size=50),
    |                ^^^^^^^^^^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1012, in test
    |     result = self.test(*args, **kwargs)
    |   File "/home/npc/pbt/agentic-pbt/worker_/7/hypo.py", line 16, in test_sparse_array_cumsum_should_not_crash
    |     result = sparse_arr.cumsum()
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
    |   File "/home/npc/pbt/agentic-pbt/worker_/7/hypo.py", line 6, in test_sparse_array_cumsum_should_not_crash
    |     data=st.lists(st.integers(min_value=-100, max_value=100), min_size=1, max_size=50),
    |                ^^^^^^^^^^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1012, in test
    |     result = self.test(*args, **kwargs)
    |   File "/home/npc/pbt/agentic-pbt/worker_/7/hypo.py", line 16, in test_sparse_array_cumsum_should_not_crash
    |     result = sparse_arr.cumsum()
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
    |   File "/home/npc/pbt/agentic-pbt/worker_/7/hypo.py", line 6, in test_sparse_array_cumsum_should_not_crash
    |     data=st.lists(st.integers(min_value=-100, max_value=100), min_size=1, max_size=50),
    |                ^^^^^^^^^^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1012, in test
    |     result = self.test(*args, **kwargs)
    |   File "/home/npc/pbt/agentic-pbt/worker_/7/hypo.py", line 16, in test_sparse_array_cumsum_should_not_crash
    |     result = sparse_arr.cumsum()
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
import pandas as pd
import numpy as np

# Create a SparseArray with fill_value=0
data = np.array([0, 1, 2])
sparse_arr = pd.arrays.SparseArray(data, fill_value=0)

print("Original SparseArray:")
print(f"  Data: {sparse_arr}")
print(f"  Fill value: {sparse_arr.fill_value}")
print(f"  Dense representation: {sparse_arr.to_dense()}")
print()

print("Attempting to call cumsum()...")
try:
    result = sparse_arr.cumsum()
    print(f"  Result: {result}")
    print(f"  Dense result: {result.to_dense()}")
except RecursionError as e:
    print(f"  RecursionError: {e}")
except Exception as e:
    print(f"  Error: {type(e).__name__}: {e}")
```

<details>

<summary>
RecursionError: maximum recursion depth exceeded
</summary>
```
Original SparseArray:
  Data: [0, 1, 2]
Fill: 0
IntIndex
Indices: array([1, 2], dtype=int32)

  Fill value: 0
  Dense representation: [0 1 2]

Attempting to call cumsum()...
  RecursionError: maximum recursion depth exceeded
```
</details>

## Why This Is A Bug

The `cumsum()` method crashes with infinite recursion for sparse arrays with non-null fill values. This affects ALL integer sparse arrays by default, since pandas infers `fill_value=0` for integer dtypes (not NaN like floats).

The bug occurs in line 1550 of `/pandas/core/arrays/sparse/array.py`:

```python
if not self._null_fill_value:
    return SparseArray(self.to_dense()).cumsum()
```

This creates an infinite recursion cycle:
1. When `fill_value` is non-null (e.g., 0 for integers), `_null_fill_value` is False
2. The code converts to dense with `self.to_dense()` returning a numpy array
3. It wraps this in `SparseArray()` without specifying a fill_value
4. For integer arrays, `SparseArray` infers `fill_value=0` by default (via `na_value_for_dtype`)
5. The new SparseArray still has `_null_fill_value=False`
6. Calling `cumsum()` on it triggers the same branch → infinite recursion

The documented behavior promises to return a SparseArray with cumulative sum, but instead the function crashes. The cumsum operation is fundamental and users expect it to work on all valid sparse arrays.

## Relevant Context

Key insights from investigation:

1. **Default fill values differ by dtype**:
   - Integer arrays: `na_value_for_dtype(int64)` returns 0 (non-null)
   - Float arrays: `na_value_for_dtype(float64)` returns NaN (null)
   - This means ALL integer sparse arrays are affected by default

2. **The recursion path**: The issue is that `SparseArray(dense_array)` for integer arrays will infer `fill_value=0` when not specified, perpetuating the non-null fill value that triggers recursion.

3. **Documentation contradiction**: The method's docstring states it returns a cumulative sum SparseArray but provides no warning about limitations with non-null fill values.

4. **Impact scope**: This affects any SparseArray where `fill_value` is not NaN/None, including the common case of integer arrays with default settings.

Relevant source code location: `/pandas/core/arrays/sparse/array.py:1550`

## Proposed Fix

The fix is to compute cumsum on the dense array before wrapping in SparseArray:

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