# Bug Report: pandas.arrays.SparseArray.cumsum() Infinite Recursion

**Target**: `pandas.arrays.SparseArray.cumsum()`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `cumsum()` method on pandas SparseArray triggers infinite recursion and crashes with a RecursionError when called on any SparseArray with a non-null fill value (such as 0 for integer arrays or any non-NaN value).

## Property-Based Test

```python
import pandas.arrays as pa
import numpy as np
from hypothesis import given, strategies as st, settings
import sys

# Set a lower recursion limit to catch the bug faster
sys.setrecursionlimit(100)

def sparse_array_strategy(min_size=0, max_size=50):
    @st.composite
    def _strat(draw):
        size = draw(st.integers(min_value=min_size, max_value=max_size))
        fill_value = draw(st.sampled_from([0, 0.0, -1, 1]))
        values = draw(st.lists(
            st.sampled_from([fill_value, 1, 2, 3, -1, 10]),
            min_size=size, max_size=size
        ))
        return pa.SparseArray(values, fill_value=fill_value)
    return _strat()


@given(sparse_array_strategy(min_size=1, max_size=20))
@settings(max_examples=100)
def test_sparsearray_cumsum_doesnt_crash(arr):
    result = arr.cumsum()
    assert isinstance(result, pa.SparseArray)

if __name__ == "__main__":
    test_sparsearray_cumsum_doesnt_crash()
```

<details>

<summary>
**Failing input**: `SparseArray([0], fill_value=0)` and many others with non-null fill values
</summary>
```
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/37/hypo.py", line 29, in <module>
  |     test_sparsearray_cumsum_doesnt_crash()
  |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  |   File "/home/npc/pbt/agentic-pbt/worker_/37/hypo.py", line 23, in test_sparsearray_cumsum_doesnt_crash
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
    |   File "/home/npc/pbt/agentic-pbt/worker_/37/hypo.py", line 23, in test_sparsearray_cumsum_doesnt_crash
    |     @settings(max_examples=100)
    |                    ^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1012, in test
    |     result = self.test(*args, **kwargs)
    |   File "/home/npc/pbt/agentic-pbt/worker_/37/hypo.py", line 25, in test_sparsearray_cumsum_doesnt_crash
    |     result = arr.cumsum()
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
    |   File "/home/npc/pbt/agentic-pbt/worker_/37/hypo.py", line 23, in test_sparsearray_cumsum_doesnt_crash
    |     @settings(max_examples=100)
    |                    ^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1012, in test
    |     result = self.test(*args, **kwargs)
    |   File "/home/npc/pbt/agentic-pbt/worker_/37/hypo.py", line 25, in test_sparsearray_cumsum_doesnt_crash
    |     result = arr.cumsum()
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
    |   File "/home/npc/pbt/agentic-pbt/worker_/37/hypo.py", line 23, in test_sparsearray_cumsum_doesnt_crash
    |     @settings(max_examples=100)
    |                    ^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1012, in test
    |     result = self.test(*args, **kwargs)
    |   File "/home/npc/pbt/agentic-pbt/worker_/37/hypo.py", line 25, in test_sparsearray_cumsum_doesnt_crash
    |     result = arr.cumsum()
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
import pandas.arrays as pa
import sys

# Set a lower recursion limit to catch the bug faster
sys.setrecursionlimit(100)

print("Testing pandas SparseArray.cumsum() with non-null fill value")
print("=" * 60)

# Create a SparseArray with fill_value=0 (non-null)
sparse = pa.SparseArray([0], fill_value=0)
print(f"Input: SparseArray([0], fill_value=0)")
print(f"Array: {sparse}")
print(f"Fill value: {sparse.fill_value}")
print()

try:
    print("Calling cumsum()...")
    result = sparse.cumsum()
    print(f"Result: {result}")
except RecursionError as e:
    print(f"RecursionError caught: {e}")
    import traceback
    print("\nTraceback (last 5 frames):")
    tb = traceback.format_exc()
    # Show only relevant parts of traceback
    lines = tb.split('\n')
    for i, line in enumerate(lines):
        if 'cumsum' in line or 'RecursionError' in line or i < 5:
            print(line)
```

<details>

<summary>
RecursionError: maximum recursion depth exceeded
</summary>
```
Testing pandas SparseArray.cumsum() with non-null fill value
============================================================
Input: SparseArray([0], fill_value=0)
Array: [0]
Fill: 0
IntIndex
Indices: array([], dtype=int32)

Fill value: 0

Calling cumsum()...
RecursionError caught: maximum recursion depth exceeded

Traceback (last 5 frames):
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/37/repo.py", line 19, in <module>
    result = sparse.cumsum()
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/sparse/array.py", line 1550, in cumsum
    return SparseArray(self.to_dense()).cumsum()
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/sparse/array.py", line 1550, in cumsum
    return SparseArray(self.to_dense()).cumsum()
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/sparse/array.py", line 1550, in cumsum
    return SparseArray(self.to_dense()).cumsum()
RecursionError: maximum recursion depth exceeded
```
</details>

## Why This Is A Bug

This is a clear programming error in the cumsum() implementation that causes infinite recursion. The bug occurs in line 1550 of `/pandas/core/arrays/sparse/array.py`:

1. **The documentation states** that cumsum() computes "Cumulative sum of non-NA/null values" and explicitly promises that "the fill value will be `np.nan` regardless" of the input fill value.

2. **The code has explicit logic** to handle non-null fill values (when `self._null_fill_value` is False), but this logic is broken. It attempts to execute:
   ```python
   return SparseArray(self.to_dense()).cumsum()
   ```
   This converts the sparse array to dense, wraps it in a new SparseArray (which inherits the same non-null fill value), and calls cumsum() again - creating infinite recursion.

3. **Integer sparse arrays default to fill_value=0**, making this a common scenario, not an edge case. Any operation on integer sparse data will trigger this crash.

4. **The method signature accepts any SparseArray** without restrictions or validation on fill_value, indicating all fill values should be supported.

5. **The crash is severe** - a complete failure with RecursionError that prevents any use of cumsum() on sparse arrays with non-null fill values.

## Relevant Context

- The `_null_fill_value` property (line 683-684) returns `True` when the fill value is NA/null (like np.nan), and `False` for non-null values like 0, -1, or 1.
- Sparse arrays with float dtype default to `fill_value=np.nan` (works correctly)
- Sparse arrays with integer dtype default to `fill_value=0` (triggers the bug)
- The bug affects pandas version 2.3.2 and likely earlier versions
- Source location: `/pandas/core/arrays/sparse/array.py:1550`

## Proposed Fix

```diff
--- a/pandas/core/arrays/sparse/array.py
+++ b/pandas/core/arrays/sparse/array.py
@@ -1547,7 +1547,7 @@ class SparseArray(OpsMixin, PandasObject, ExtensionArray):
             raise ValueError(f"axis(={axis}) out of bounds")

         if not self._null_fill_value:
-            return SparseArray(self.to_dense()).cumsum()
+            return SparseArray(self.to_dense().cumsum(), fill_value=np.nan)

         return SparseArray(
             self.sp_values.cumsum(),
```