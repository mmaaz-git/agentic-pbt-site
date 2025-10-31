# Bug Report: pandas.core.arrays.sparse.SparseArray.argmax/argmin crash on all-fill arrays

**Target**: `pandas.core.arrays.sparse.SparseArray.argmax()` and `pandas.core.arrays.sparse.SparseArray.argmin()`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

When calling `argmax()` or `argmin()` on a SparseArray where all values equal the fill_value, the methods crash with `ValueError: attempt to get argmax of an empty sequence` instead of returning the index of the first element.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import numpy as np
from pandas.core.arrays.sparse import SparseArray

@given(
    data=st.lists(st.integers(min_value=-1000, max_value=1000), min_size=1, max_size=100),
    fill_value=st.integers(min_value=-1000, max_value=1000)
)
@settings(max_examples=1000, deadline=None)
def test_sparse_array_argmax_argmin_match_dense(data, fill_value):
    """Test that SparseArray.argmax/argmin matches NumPy array behavior"""
    arr = np.array(data)
    sparse = SparseArray(arr, fill_value=fill_value)

    # Test argmax
    try:
        sparse_argmax = sparse.argmax()
        numpy_argmax = arr.argmax()
        assert sparse_argmax == numpy_argmax, f"argmax mismatch: sparse={sparse_argmax}, numpy={numpy_argmax}"
    except Exception as e:
        print(f"argmax failed with data={data}, fill_value={fill_value}")
        raise

    # Test argmin
    try:
        sparse_argmin = sparse.argmin()
        numpy_argmin = arr.argmin()
        assert sparse_argmin == numpy_argmin, f"argmin mismatch: sparse={sparse_argmin}, numpy={numpy_argmin}"
    except Exception as e:
        print(f"argmin failed with data={data}, fill_value={fill_value}")
        raise

if __name__ == "__main__":
    # Run the test
    test_sparse_array_argmax_argmin_match_dense()
    print("All tests passed!")
```

<details>

<summary>
**Failing input**: `data=[0], fill_value=0`
</summary>
```
argmax failed with data=[0], fill_value=0
argmax failed with data=[0], fill_value=0
argmax failed with data=[0], fill_value=0
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/40/hypo.py", line 35, in <module>
    test_sparse_array_argmax_argmin_match_dense()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/40/hypo.py", line 6, in test_sparse_array_argmax_argmin_match_dense
    data=st.lists(st.integers(min_value=-1000, max_value=1000), min_size=1, max_size=100),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/40/hypo.py", line 17, in test_sparse_array_argmax_argmin_match_dense
    sparse_argmax = sparse.argmax()
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/sparse/array.py", line 1678, in argmax
    return self._argmin_argmax("argmax")
           ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/sparse/array.py", line 1658, in _argmin_argmax
    _candidate = non_nan_idx[func(non_nans)]
                             ~~~~^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/fromnumeric.py", line 1341, in argmax
    return _wrapfunc(a, 'argmax', axis=axis, out=out, **kwds)
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/fromnumeric.py", line 57, in _wrapfunc
    return bound(*args, **kwds)
ValueError: attempt to get argmax of an empty sequence
Falsifying example: test_sparse_array_argmax_argmin_match_dense(
    data=[0],
    fill_value=0,
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/fromnumeric.py:58
```
</details>

## Reproducing the Bug

```python
import numpy as np
from pandas.core.arrays.sparse import SparseArray

# Test case with all values equal to fill_value
data = [0]
sparse = SparseArray(data, fill_value=0)

print("Creating SparseArray with data=[0] and fill_value=0")
print(f"SparseArray: {sparse}")
print(f"sp_values: {sparse.sp_values}")
print(f"fill_value: {sparse.fill_value}")

print("\nCalling argmax()...")
try:
    result = sparse.argmax()
    print(f"argmax() result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print("\nCalling argmin()...")
try:
    result = sparse.argmin()
    print(f"argmin() result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

# Compare with NumPy behavior
print("\nNumPy behavior for comparison:")
arr = np.array([0])
print(f"np.array([0]).argmax() = {arr.argmax()}")
print(f"np.array([0]).argmin() = {arr.argmin()}")
```

<details>

<summary>
ValueError: attempt to get argmax of an empty sequence
</summary>
```
Creating SparseArray with data=[0] and fill_value=0
SparseArray: [0]
Fill: 0
IntIndex
Indices: array([], dtype=int32)

sp_values: []
fill_value: 0

Calling argmax()...
Error: ValueError: attempt to get argmax of an empty sequence

Calling argmin()...
Error: ValueError: attempt to get argmin of an empty sequence

NumPy behavior for comparison:
np.array([0]).argmax() = 0
np.array([0]).argmin() = 0
```
</details>

## Why This Is A Bug

This violates expected behavior because `argmax()` and `argmin()` should return the index of the maximum/minimum value in the array, consistent with NumPy's behavior. When all values are equal (including the case where all values equal the fill_value), NumPy returns 0 (the first index), but SparseArray crashes instead.

The crash occurs because:
1. When all array values equal the fill_value, the sparse representation stores no non-fill values, so `sp_values` is empty
2. The `_argmin_argmax` method (line 1648 in `/pandas/core/arrays/sparse/array.py`) attempts to find the argmax/argmin of these sparse values
3. After filtering out NaN values, it calls `func(non_nans)` on line 1658, where `non_nans` is empty
4. NumPy's `argmax()` and `argmin()` raise a ValueError when called on an empty array
5. This error is not caught, causing the method to crash rather than handling this edge case

This is inconsistent with NumPy arrays where `np.array([0]).argmax()` returns `0`, not an error. The SparseArray implementation should handle the case where all values are fill values and return an appropriate index (typically 0 for the first element).

## Relevant Context

The bug is located in the `_argmin_argmax` method at line 1648-1672 of `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/core/arrays/sparse/array.py`. The method doesn't handle the edge case where `sp_values` is empty (all values are fill_value).

Key observations:
- The sparse representation is working correctly - when all values equal fill_value, storing no sparse values is the expected optimization
- The issue is purely in the argmax/argmin implementation not handling this valid sparse state
- The fix should ensure consistency with NumPy behavior, where argmax/argmin of an array with all equal values returns the first index

## Proposed Fix

```diff
--- a/pandas/core/arrays/sparse/array.py
+++ b/pandas/core/arrays/sparse/array.py
@@ -1654,6 +1654,11 @@ class SparseArray(OpsMixin, PandasObject, ExtensionArray):
         idx = np.arange(values.shape[0])
         non_nans = values[~mask]
         non_nan_idx = idx[~mask]
+
+        # Handle case where all values are fill_value (empty sparse values)
+        if len(non_nans) == 0:
+            # Return first fill_value location if it exists, otherwise 0
+            return self._first_fill_value_loc() if self._first_fill_value_loc() != -1 else 0

         _candidate = non_nan_idx[func(non_nans)]
         candidate = index[_candidate]
```