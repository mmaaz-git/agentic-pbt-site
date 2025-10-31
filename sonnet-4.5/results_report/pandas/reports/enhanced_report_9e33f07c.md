# Bug Report: pandas.core.arrays.sparse.SparseArray.cumsum Infinite Recursion with Non-Null Fill Values

**Target**: `pandas.core.arrays.sparse.SparseArray.cumsum`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `cumsum()` method on `SparseArray` enters infinite recursion when called on arrays with non-null fill values, causing a `RecursionError` and crashing the application.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from pandas.arrays import SparseArray

@st.composite
def sparse_arrays(draw, min_size=0, max_size=100):
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    dtype_choice = draw(st.sampled_from(['int64', 'bool']))

    if dtype_choice == 'int64':
        values = draw(st.lists(st.integers(min_value=-1000, max_value=1000),
                              min_size=size, max_size=size))
        fill_value = 0
    else:
        values = draw(st.lists(st.booleans(), min_size=size, max_size=size))
        fill_value = False

    kind = draw(st.sampled_from(['integer', 'block']))
    return SparseArray(values, fill_value=fill_value, kind=kind)

@given(sparse_arrays())
@settings(max_examples=100)
def test_cumsum_length(arr):
    """Cumsum should preserve length"""
    cumsum_result = arr.cumsum()
    assert len(cumsum_result) == len(arr)

if __name__ == "__main__":
    test_cumsum_length()
```

<details>

<summary>
**Failing input**: `SparseArray([True], fill_value=False)`
</summary>
```
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/30/hypo.py", line 28, in <module>
  |     test_cumsum_length()
  |     ~~~~~~~~~~~~~~~~~~^^
  |   File "/home/npc/pbt/agentic-pbt/worker_/30/hypo.py", line 21, in test_cumsum_length
  |     @settings(max_examples=100)
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
    |   File "/home/npc/pbt/agentic-pbt/worker_/30/hypo.py", line 21, in test_cumsum_length
    |     @settings(max_examples=100)
    |                    ^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1012, in test
    |     result = self.test(*args, **kwargs)
    |   File "/home/npc/pbt/agentic-pbt/worker_/30/hypo.py", line 24, in test_cumsum_length
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
    |   File "/home/npc/pbt/agentic-pbt/worker_/30/hypo.py", line 21, in test_cumsum_length
    |     @settings(max_examples=100)
    |                    ^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1012, in test
    |     result = self.test(*args, **kwargs)
    |   File "/home/npc/pbt/agentic-pbt/worker_/30/hypo.py", line 24, in test_cumsum_length
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
    |   File "/home/npc/pbt/agentic-pbt/worker_/30/hypo.py", line 21, in test_cumsum_length
    |     @settings(max_examples=100)
    |                    ^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1012, in test
    |     result = self.test(*args, **kwargs)
    |   File "/home/npc/pbt/agentic-pbt/worker_/30/hypo.py", line 24, in test_cumsum_length
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
import sys

# Set a lower recursion limit to make the error happen faster
sys.setrecursionlimit(50)

# Create a boolean sparse array (which cannot have NaN fill values)
arr = SparseArray([True], fill_value=False)
print(f"Array: {arr}")
print(f"Array type: {type(arr)}")
print(f"Array dtype: {arr.dtype}")
print(f"Fill value: {arr.fill_value}")
print(f"_null_fill_value: {arr._null_fill_value}")
print()
print("Calling cumsum()...")

try:
    result = arr.cumsum()
    print(f"Result: {result}")
except RecursionError as e:
    print(f"RecursionError: {e}")
```

<details>

<summary>
RecursionError when calling cumsum() on boolean SparseArray
</summary>
```
Array: [True]
Fill: False
IntIndex
Indices: array([0], dtype=int32)

Array type: <class 'pandas.core.arrays.sparse.array.SparseArray'>
Array dtype: Sparse[bool, False]
Fill value: False
_null_fill_value: False

Calling cumsum()...
RecursionError: maximum recursion depth exceeded
```
</details>

## Why This Is A Bug

This violates expected behavior because `cumsum()` is a standard array operation that should work on all valid SparseArray instances. The implementation contains a logic error that causes infinite recursion.

The bug occurs in the cumsum method at line 1550 of `/pandas/core/arrays/sparse/array.py`. When `_null_fill_value` is False (which happens for boolean arrays and integer arrays with non-NaN fill values), the code executes:

```python
if not self._null_fill_value:
    return SparseArray(self.to_dense()).cumsum()
```

This creates infinite recursion because:
1. `self.to_dense()` correctly converts to a NumPy array
2. `SparseArray(...)` wraps it back into a SparseArray with the same non-null fill_value
3. `.cumsum()` is called on this new SparseArray
4. The condition `not self._null_fill_value` is still True (since the fill_value hasn't changed)
5. The cycle repeats infinitely

The code clearly intends to handle non-null fill values (as evidenced by the specific branch for this condition), but implements it incorrectly.

## Relevant Context

This affects all sparse arrays where the fill_value cannot be NaN:
- **All boolean sparse arrays** - Boolean dtype cannot have NaN values by definition, so `fill_value` must be either True or False
- **Integer sparse arrays with explicit fill values** - e.g., `SparseArray([1, 2, 3], fill_value=0)`
- **Float sparse arrays with non-NaN fill values** - e.g., `SparseArray([1.0, 2.0], fill_value=0.0)`

The `_null_fill_value` property is defined in `/pandas/core/dtypes/dtypes.py:1787` and returns `True` only when `isna(self.fill_value)` is true.

A workaround exists: users can manually convert to dense, apply cumsum, then convert back:
```python
result = SparseArray(arr.to_dense().cumsum())
```

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