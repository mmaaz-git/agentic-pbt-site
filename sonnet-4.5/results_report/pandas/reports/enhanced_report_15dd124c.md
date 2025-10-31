# Bug Report: pandas.core.arrays.sparse.SparseArray.cumsum Infinite Recursion

**Target**: `pandas.core.arrays.sparse.SparseArray.cumsum`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `cumsum()` method on SparseArray causes infinite recursion and crashes with RecursionError when the array has a non-null fill value, which is the default behavior for all integer SparseArrays (fill_value=0).

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy as np
from pandas.arrays import SparseArray

@given(st.lists(st.integers(min_value=-100, max_value=100), min_size=1, max_size=100))
def test_cumsum_does_not_crash(values):
    arr = np.array(values)
    sparse = SparseArray(arr)

    cumsum_result = sparse.cumsum()
    assert len(cumsum_result) == len(sparse)

# Run the test
if __name__ == "__main__":
    test_cumsum_does_not_crash()
```

<details>

<summary>
**Failing input**: `[0]` (any integer list triggers the error)
</summary>
```
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/57/hypo.py", line 15, in <module>
  |     test_cumsum_does_not_crash()
  |     ~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  |   File "/home/npc/pbt/agentic-pbt/worker_/57/hypo.py", line 6, in test_cumsum_does_not_crash
  |     def test_cumsum_does_not_crash(values):
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
    |   File "/home/npc/pbt/agentic-pbt/worker_/57/hypo.py", line 6, in test_cumsum_does_not_crash
    |     def test_cumsum_does_not_crash(values):
    |                    ^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1012, in test
    |     result = self.test(*args, **kwargs)
    |   File "/home/npc/pbt/agentic-pbt/worker_/57/hypo.py", line 10, in test_cumsum_does_not_crash
    |     cumsum_result = sparse.cumsum()
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
    |   File "/home/npc/pbt/agentic-pbt/worker_/57/hypo.py", line 6, in test_cumsum_does_not_crash
    |     def test_cumsum_does_not_crash(values):
    |                    ^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1012, in test
    |     result = self.test(*args, **kwargs)
    |   File "/home/npc/pbt/agentic-pbt/worker_/57/hypo.py", line 10, in test_cumsum_does_not_crash
    |     cumsum_result = sparse.cumsum()
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
    |   File "/home/npc/pbt/agentic-pbt/worker_/57/hypo.py", line 6, in test_cumsum_does_not_crash
    |     def test_cumsum_does_not_crash(values):
    |                    ^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1012, in test
    |     result = self.test(*args, **kwargs)
    |   File "/home/npc/pbt/agentic-pbt/worker_/57/hypo.py", line 10, in test_cumsum_does_not_crash
    |     cumsum_result = sparse.cumsum()
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
import numpy as np

# Create a simple SparseArray with integer values
# Integer arrays default to fill_value=0
sparse = SparseArray([1, 2, 3])

print(f"SparseArray: {sparse}")
print(f"Fill value: {sparse.fill_value}")
print(f"_null_fill_value: {sparse._null_fill_value}")

# This should calculate cumulative sum [1, 3, 6]
# But it will cause RecursionError
try:
    result = sparse.cumsum()
    print(f"Cumsum result: {result}")
except RecursionError as e:
    print(f"\nRecursionError occurred!")
    print(f"Error: {e}")
```

<details>

<summary>
RecursionError: maximum recursion depth exceeded
</summary>
```
SparseArray: [1, 2, 3]
Fill: 0
IntIndex
Indices: array([0, 1, 2], dtype=int32)

Fill value: 0
_null_fill_value: False

RecursionError occurred!
Error: maximum recursion depth exceeded
```
</details>

## Why This Is A Bug

The cumsum() method is expected to compute the cumulative sum of array elements. For the input [1, 2, 3], it should return [1, 3, 6]. Instead, it crashes with infinite recursion.

The root cause is in the implementation at pandas/core/arrays/sparse/array.py lines 1549-1550:

```python
if not self._null_fill_value:
    return SparseArray(self.to_dense()).cumsum()
```

When a SparseArray has a non-null fill value (e.g., fill_value=0 for integer arrays, which is the default):
1. The condition `not self._null_fill_value` evaluates to True
2. The code converts the sparse array to a dense array with `self.to_dense()`
3. It creates a NEW SparseArray from the dense array
4. This new SparseArray inherits the same dtype and fill_value (0 for integers)
5. Therefore, the new SparseArray also has `_null_fill_value=False`
6. Calling `cumsum()` on it enters the same conditional branch
7. This creates another SparseArray and calls cumsum() again, causing infinite recursion

This bug affects ALL integer SparseArrays because they default to fill_value=0. It also affects any SparseArray explicitly created with a non-NA fill value. The cumsum operation is a fundamental array operation that should work on all valid inputs.

## Relevant Context

- **Affected versions**: Current pandas implementation
- **Default behavior**: When creating a SparseArray from integer data, pandas automatically sets fill_value=0
- **Working case**: SparseArrays with NaN fill_value (e.g., float arrays) work correctly because `_null_fill_value` is True for NaN
- **Impact**: This makes cumsum completely unusable for the most common case of integer sparse arrays
- **Documentation**: The cumsum docstring mentions handling NA/null values but doesn't document this crash behavior

The `_null_fill_value` property (line 683-684) returns `self._dtype._is_na_fill_value`, which checks if the fill value is NA/NaN. For integer arrays with fill_value=0, this returns False, triggering the buggy code path.

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