# Bug Report: pandas.core.arrays.sparse.SparseArray argmin/argmax Methods Crash When All Values Equal Fill Value

**Target**: `pandas.core.arrays.sparse.array.SparseArray.argmin` and `pandas.core.arrays.sparse.array.SparseArray.argmax`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `argmin()` and `argmax()` methods on `SparseArray` crash with `ValueError: attempt to get argmin of an empty sequence` when all elements in the array equal the fill value, instead of returning a valid index as expected.

## Property-Based Test

```python
from pandas.arrays import SparseArray
from hypothesis import given, strategies as st, settings

@given(
    st.lists(st.integers(min_value=-100, max_value=100), min_size=2, max_size=50)
)
@settings(max_examples=500)
def test_argmin_argmax_values(data):
    arr = SparseArray(data)

    argmin_idx = arr.argmin()
    argmax_idx = arr.argmax()

    assert arr[argmin_idx] == arr.min()
    assert arr[argmax_idx] == arr.max()

if __name__ == "__main__":
    test_argmin_argmax_values()
```

<details>

<summary>
**Failing input**: `[0, 0]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/4/hypo.py", line 18, in <module>
    test_argmin_argmax_values()
    ~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/4/hypo.py", line 5, in test_argmin_argmax_values
    st.lists(st.integers(min_value=-100, max_value=100), min_size=2, max_size=50)
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/4/hypo.py", line 11, in test_argmin_argmax_values
    argmin_idx = arr.argmin()
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
Falsifying example: test_argmin_argmax_values(
    data=[0, 0],
)
```
</details>

## Reproducing the Bug

```python
from pandas.arrays import SparseArray

# Test case with all values equal to the fill value
arr = SparseArray([0, 0, 0], fill_value=0)

print("Testing argmin on SparseArray([0, 0, 0], fill_value=0):")
try:
    result = arr.argmin()
    print(f"argmin() returned: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print("\nTesting argmax on SparseArray([0, 0, 0], fill_value=0):")
try:
    result = arr.argmax()
    print(f"argmax() returned: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
```

<details>

<summary>
ValueError: attempt to get argmin/argmax of an empty sequence
</summary>
```
Testing argmin on SparseArray([0, 0, 0], fill_value=0):
Error: ValueError: attempt to get argmin of an empty sequence

Testing argmax on SparseArray([0, 0, 0], fill_value=0):
Error: ValueError: attempt to get argmax of an empty sequence
```
</details>

## Why This Is A Bug

This violates the expected behavior of `argmin()` and `argmax()` methods which should always return an integer index for non-empty arrays, as documented in pandas. The methods are documented to return `int` - the index of the minimum/maximum element.

For an array like `[0, 0, 0]`, both `argmin()` and `argmax()` should return `0` (the first index), consistent with NumPy's behavior:
- `numpy.array([0, 0, 0]).argmin()` returns `0`
- `numpy.array([0, 0, 0]).argmax()` returns `0`

The crash occurs because SparseArrays store only non-fill values in `_sparse_values`. When all values equal the fill value (e.g., all zeros with `fill_value=0`), the `_sparse_values` array is empty. The code at line 1658 in `/pandas/core/arrays/sparse/array.py` then attempts to call `np.argmin()` or `np.argmax()` on this empty array, causing the ValueError.

## Relevant Context

The bug is in the `_argmin_argmax` method at lines 1648-1672 of `pandas/core/arrays/sparse/array.py`. The method doesn't handle the case where `self._sparse_values` is empty (when `npoints == 0`).

Key points:
- SparseArrays use `fill_value` (default 0) to represent common values efficiently
- Only values different from `fill_value` are stored in `_sparse_values`
- When all array elements equal `fill_value`, `_sparse_values` is empty
- The method calls numpy's argmin/argmax on the empty `_sparse_values`, causing the crash

Documentation link: https://pandas.pydata.org/docs/reference/api/pandas.arrays.SparseArray.argmin.html

## Proposed Fix

```diff
--- a/pandas/core/arrays/sparse/array.py
+++ b/pandas/core/arrays/sparse/array.py
@@ -1655,6 +1655,11 @@ class SparseArray(OpsMixin, PandasObject, ExtensionArray):
         non_nans = values[~mask]
         non_nan_idx = idx[~mask]

+        # Handle case where all values are fill_value (empty sparse values)
+        if len(non_nans) == 0:
+            # Return first valid index when all values are the same
+            return 0
+
         _candidate = non_nan_idx[func(non_nans)]
         candidate = index[_candidate]
```