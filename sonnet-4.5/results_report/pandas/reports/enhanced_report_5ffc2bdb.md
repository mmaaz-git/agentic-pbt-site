# Bug Report: pandas.core.arrays.sparse.SparseArray argmin/argmax Empty Sequence Crash

**Target**: `pandas.core.arrays.sparse.SparseArray.argmin` and `pandas.core.arrays.sparse.SparseArray.argmax`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

SparseArray's argmin() and argmax() methods crash with a ValueError when the array contains only fill_value elements, causing numpy to attempt argmin/argmax on an empty sequence.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy as np
from pandas.arrays import SparseArray

@given(st.lists(st.integers(min_value=-100, max_value=100), min_size=1, max_size=100))
def test_argmin_argmax_no_crash(values):
    arr = np.array(values)
    sparse = SparseArray(arr)

    argmin_result = sparse.argmin()
    argmax_result = sparse.argmax()

    assert argmin_result == arr.argmin()
    assert argmax_result == arr.argmax()

# Run the test
if __name__ == "__main__":
    test_argmin_argmax_no_crash()
```

<details>

<summary>
**Failing input**: `values=[0]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/34/hypo.py", line 18, in <module>
    test_argmin_argmax_no_crash()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/34/hypo.py", line 6, in test_argmin_argmax_no_crash
    def test_argmin_argmax_no_crash(values):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/34/hypo.py", line 10, in test_argmin_argmax_no_crash
    argmin_result = sparse.argmin()
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/sparse/array.py", line 1684, in argmin
    return self._argmin_argmax("argmin")
           ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/sparse/array.py", line 1658, in _argmin_argmax
    _candidate = non_nan_idx[func(non_nans)]
                             ~~~~^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/fromnumeric.py", line 1439, in argmin
    return _wrapfunc(a, 'argmin', axis=axis, out=out, **kwds)
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/fromnumeric.py", line 57, in _wrapfunc
    return bound(*args, **kwds)
ValueError: attempt to get argmin of an empty sequence
Falsifying example: test_argmin_argmax_no_crash(
    values=[0],
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/fromnumeric.py:58
```
</details>

## Reproducing the Bug

```python
from pandas.arrays import SparseArray
import numpy as np

# Test case that should crash
sparse = SparseArray([0])  # Array containing only fill_value (default is 0)

print("Testing SparseArray([0]).argmin()...")
result = sparse.argmin()
print(f"Result: {result}")
```

<details>

<summary>
ValueError: attempt to get argmin of an empty sequence
</summary>
```
Testing SparseArray([0]).argmin()...
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/34/repo.py", line 8, in <module>
    result = sparse.argmin()
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/sparse/array.py", line 1684, in argmin
    return self._argmin_argmax("argmin")
           ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/sparse/array.py", line 1658, in _argmin_argmax
    _candidate = non_nan_idx[func(non_nans)]
                             ~~~~^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/fromnumeric.py", line 1439, in argmin
    return _wrapfunc(a, 'argmin', axis=axis, out=out, **kwds)
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/fromnumeric.py", line 57, in _wrapfunc
    return bound(*args, **kwds)
ValueError: attempt to get argmin of an empty sequence
```
</details>

## Why This Is A Bug

This violates expected behavior because SparseArray should behave consistently with numpy arrays for basic operations like argmin/argmax. When a numpy array contains all identical values, `np.array([0]).argmin()` returns 0 (the first index), not an error.

The bug occurs in the `_argmin_argmax` method at line 1658 of `/pandas/core/arrays/sparse/array.py`. When a SparseArray contains only fill_value elements (like [0] with default fill_value=0), the sparse representation stores no actual values in `sp_values`. This makes `non_nans` empty after filtering, causing numpy's argmin/argmax to fail with "ValueError: attempt to get argmin of an empty sequence".

SparseArray is designed to be a drop-in replacement for regular arrays in pandas operations, so this inconsistency breaks that contract. The method should handle the case where all elements equal the fill_value by returning index 0, matching numpy's behavior for arrays with all identical values.

## Relevant Context

- **SparseArray internals**: SparseArray uses a sparse representation where only non-fill values are stored in `sp_values`. When all values equal `fill_value`, `sp_values` is empty.
- **Default fill_value**: The default fill_value for SparseArray is 0, making `SparseArray([0])` a common edge case.
- **NumPy behavior**: `np.array([0]).argmin()` returns 0 without error, establishing the expected behavior.
- **Common occurrence**: Arrays initialized to all zeros or containing uniform values are common in data processing pipelines.
- **Code location**: The bug is in `/pandas/core/arrays/sparse/array.py` at line 1658 in the `_argmin_argmax` method.

## Proposed Fix

```diff
--- a/pandas/core/arrays/sparse/array.py
+++ b/pandas/core/arrays/sparse/array.py
@@ -1654,8 +1654,13 @@ class SparseArray(OpsMixin, PandasObject, ExtensionArray):
         idx = np.arange(values.shape[0])
         non_nans = values[~mask]
         non_nan_idx = idx[~mask]

+        if len(non_nans) == 0:
+            # All values are fill_value, return first index
+            # This matches numpy behavior for arrays with all identical values
+            return 0
+
         _candidate = non_nan_idx[func(non_nans)]
         candidate = index[_candidate]

         if isna(self.fill_value):
```