# Bug Report: pandas.arrays.SparseArray.cumsum Infinite Recursion

**Target**: `pandas.arrays.SparseArray.cumsum`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `cumsum()` method on `SparseArray` with non-null fill_value triggers infinite recursion, causing a RecursionError crash on any input.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import pandas.arrays

@given(st.lists(st.integers(min_value=0, max_value=100), min_size=1, max_size=30))
@settings(max_examples=500)
def test_sparse_array_cumsum_length(data):
    sparse = pandas.arrays.SparseArray(data, fill_value=0)
    cumsum_result = sparse.cumsum()
    assert len(cumsum_result) == len(sparse)

# Run the test
if __name__ == "__main__":
    test_sparse_array_cumsum_length()
```

<details>

<summary>
**Failing input**: `data=[0]`
</summary>
```
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/49/hypo.py", line 13, in <module>
  |     test_sparse_array_cumsum_length()
  |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  |   File "/home/npc/pbt/agentic-pbt/worker_/49/hypo.py", line 5, in test_sparse_array_cumsum_length
  |     @settings(max_examples=500)
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
    |   File "/home/npc/pbt/agentic-pbt/worker_/49/hypo.py", line 5, in test_sparse_array_cumsum_length
    |     @settings(max_examples=500)
    |                    ^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1012, in test
    |     result = self.test(*args, **kwargs)
    |   File "/home/npc/pbt/agentic-pbt/worker_/49/hypo.py", line 8, in test_sparse_array_cumsum_length
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
    |   File "/home/npc/pbt/agentic-pbt/worker_/49/hypo.py", line 5, in test_sparse_array_cumsum_length
    |     @settings(max_examples=500)
    |                    ^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1012, in test
    |     result = self.test(*args, **kwargs)
    |   File "/home/npc/pbt/agentic-pbt/worker_/49/hypo.py", line 8, in test_sparse_array_cumsum_length
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
    |   File "/home/npc/pbt/agentic-pbt/worker_/49/hypo.py", line 5, in test_sparse_array_cumsum_length
    |     @settings(max_examples=500)
    |                    ^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1012, in test
    |     result = self.test(*args, **kwargs)
    |   File "/home/npc/pbt/agentic-pbt/worker_/49/hypo.py", line 8, in test_sparse_array_cumsum_length
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
import pandas.arrays
import numpy as np

# Test case 1: Basic case with fill_value=0
print("Test 1: Basic case with fill_value=0")
print("Creating SparseArray([1, 0, 2, 0, 3], fill_value=0)")
try:
    sparse = pandas.arrays.SparseArray([1, 0, 2, 0, 3], fill_value=0)
    print(f"SparseArray created: {sparse}")
    print("Calling cumsum()...")
    result = sparse.cumsum()
    print(f"Result: {result}")
except RecursionError as e:
    print(f"RecursionError: {e}")
    import traceback
    print("\nPartial traceback (showing key recursive calls):")
    tb = traceback.format_exc()
    lines = tb.split('\n')
    # Show first few and last few lines to see the recursion pattern
    for i, line in enumerate(lines[:10]):
        print(line)
    print("...")
    for line in lines[-10:]:
        print(line)

print("\n" + "="*60 + "\n")

# Test case 2: Works fine with fill_value=np.nan
print("Test 2: Case with fill_value=np.nan (should work)")
print("Creating SparseArray([1, np.nan, 2, np.nan, 3], fill_value=np.nan)")
try:
    sparse_nan = pandas.arrays.SparseArray([1, np.nan, 2, np.nan, 3], fill_value=np.nan)
    print(f"SparseArray created: {sparse_nan}")
    print("Calling cumsum()...")
    result_nan = sparse_nan.cumsum()
    print(f"Result: {result_nan}")
    print("Success! No recursion error with NaN fill_value.")
except RecursionError as e:
    print(f"RecursionError: {e}")
```

<details>

<summary>
RecursionError: maximum recursion depth exceeded
</summary>
```
Test 1: Basic case with fill_value=0
Creating SparseArray([1, 0, 2, 0, 3], fill_value=0)
SparseArray created: [1, 0, 2, 0, 3]
Fill: 0
IntIndex
Indices: array([0, 2, 4], dtype=int32)

Calling cumsum()...
RecursionError: maximum recursion depth exceeded

Partial traceback (showing key recursive calls):
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/49/repo.py", line 11, in <module>
    result = sparse.cumsum()
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/sparse/array.py", line 1550, in cumsum
    return SparseArray(self.to_dense()).cumsum()
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/sparse/array.py", line 1550, in cumsum
    return SparseArray(self.to_dense()).cumsum()
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/sparse/array.py", line 1550, in cumsum
...
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/dtypes/dtypes.py", line 1777, in _check_fill_value
    if not can_hold_element(dummy, val):
           ~~~~~~~~~~~~~~~~^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/dtypes/cast.py", line 1772, in can_hold_element
    np_can_hold_element(dtype, element)
    ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/dtypes/cast.py", line 1810, in np_can_hold_element
    info = np.iinfo(dtype)
RecursionError: maximum recursion depth exceeded


============================================================

Test 2: Case with fill_value=np.nan (should work)
Creating SparseArray([1, np.nan, 2, np.nan, 3], fill_value=np.nan)
SparseArray created: [1.0, nan, 2.0, nan, 3.0]
Fill: nan
IntIndex
Indices: array([0, 2, 4], dtype=int32)

Calling cumsum()...
Result: [1.0, nan, 3.0, nan, 6.0]
Fill: nan
IntIndex
Indices: array([0, 2, 4], dtype=int32)

Success! No recursion error with NaN fill_value.
```
</details>

## Why This Is A Bug

The implementation at line 1550 of `/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/sparse/array.py` contains a critical parenthesis placement error:

```python
return SparseArray(self.to_dense()).cumsum()
```

This code creates infinite recursion by:
1. Converting the sparse array to dense: `self.to_dense()` returns a numpy array
2. Wrapping it in a new SparseArray: `SparseArray(...)` creates a new sparse array
3. Calling `.cumsum()` on the new SparseArray, which triggers the same code path again
4. This repeats indefinitely until Python's recursion limit is reached

The method docstring explicitly states it should compute "Cumulative sum of non-NA/null values" and return a SparseArray. The code has two branches:
- When `_null_fill_value` is True (fill_value is NaN): Works correctly using sparse values directly
- When `_null_fill_value` is False (fill_value is 0, etc.): Triggers infinite recursion

The intent is clearly to compute cumsum on the dense array BEFORE wrapping it back in SparseArray.

## Relevant Context

The `_null_fill_value` property determines whether the fill_value is considered null/NaN. When False (for values like 0), the implementation needs to convert to dense format because the cumsum calculation must account for the fill values in the computation.

Code location: `/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/sparse/array.py:1550`

The bug affects any SparseArray with non-null fill_value, which is extremely common in practice (e.g., using 0 for sparse matrices). The workaround requires manually implementing what the method should do: `SparseArray(sparse.to_dense().cumsum())`.

## Proposed Fix

```diff
--- a/pandas/core/arrays/sparse/array.py
+++ b/pandas/core/arrays/sparse/array.py
@@ -1547,7 +1547,7 @@ class SparseArray(OpsMixin, ExtensionArray):
             raise ValueError(f"axis(={axis}) out of bounds")

         if not self._null_fill_value:
-            return SparseArray(self.to_dense()).cumsum()
+            return SparseArray(self.to_dense().cumsum())

         return SparseArray(
             self.sp_values.cumsum(),
```