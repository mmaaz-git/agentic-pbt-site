# Bug Report: pandas.arrays.SparseArray argmin/argmax Crash on All Fill Values

**Target**: `pandas.core.arrays.sparse.array.SparseArray.argmin` and `pandas.core.arrays.sparse.array.SparseArray.argmax`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `argmin()` and `argmax()` methods crash with `ValueError: attempt to get argmin of an empty sequence` when called on a SparseArray where all elements equal the fill value, instead of returning a valid index as numpy and pandas conventions dictate.

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
  File "/home/npc/pbt/agentic-pbt/worker_/11/hypo.py", line 18, in <module>
    test_argmin_argmax_values()
    ~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/11/hypo.py", line 5, in test_argmin_argmax_values
    st.lists(st.integers(min_value=-100, max_value=100), min_size=2, max_size=50)
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/11/hypo.py", line 11, in test_argmin_argmax_values
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

# Test case that should crash with all fill values
arr = SparseArray([0, 0, 0], fill_value=0)
print(f"Created SparseArray: {arr}")
print(f"Array values: {arr.to_numpy()}")
print(f"Fill value: {arr.fill_value}")

try:
    result = arr.argmin()
    print(f"argmin() returned: {result}")
except Exception as e:
    print(f"argmin() crashed with {type(e).__name__}: {e}")

try:
    result = arr.argmax()
    print(f"argmax() returned: {result}")
except Exception as e:
    print(f"argmax() crashed with {type(e).__name__}: {e}")
```

<details>

<summary>
ValueError crash for both argmin() and argmax()
</summary>
```
Created SparseArray: [0, 0, 0]
Fill: 0
IntIndex
Indices: array([], dtype=int32)

Array values: [0 0 0]
Fill value: 0
argmin() crashed with ValueError: attempt to get argmin of an empty sequence
argmax() crashed with ValueError: attempt to get argmax of an empty sequence
```
</details>

## Why This Is A Bug

This violates expected behavior in multiple ways:

1. **Inconsistent with NumPy convention**: NumPy's `argmin()` and `argmax()` return index 0 for arrays with all identical values. For `np.array([0, 0, 0])`, both `argmin()` and `argmax()` correctly return 0.

2. **Inconsistent with pandas' own methods**: The same SparseArray's `min()` and `max()` methods work correctly, returning the fill value (0 in this case). If min/max can handle all-fill-value arrays, argmin/argmax should too.

3. **Violates documented contract**: The docstring for `argmin()` states: "Return the index of minimum value. In case of multiple occurrences of the minimum value, the index corresponding to the first occurrence is returned." An array of all zeros has multiple occurrences of the minimum, so it should return 0, not crash.

4. **Breaks on valid data**: SparseArrays with all values equal to the fill value are valid and can occur naturally in data processing (e.g., sparse matrices with an all-zero row/column, boolean masks that are all False, etc.).

## Relevant Context

The crash occurs in `_argmin_argmax()` at line 1658 of `/pandas/core/arrays/sparse/array.py`:

```python
_candidate = non_nan_idx[func(non_nans)]
```

When all values equal the fill value, the SparseArray's internal `_sparse_values` array is empty (sparse storage only stores non-fill values). This makes `non_nans` an empty array after filtering, and calling `np.argmin()` or `np.argmax()` on an empty array raises the ValueError.

The method already has logic to handle fill values (lines 1661-1672) that compares the candidate sparse value with the fill value and returns the appropriate index. However, this code is never reached when there are no sparse values.

**Documentation reference**: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.arrays.SparseArray.argmin.html

**Code location**: pandas/core/arrays/sparse/array.py:1648-1684

## Proposed Fix

```diff
--- a/pandas/core/arrays/sparse/array.py
+++ b/pandas/core/arrays/sparse/array.py
@@ -1655,6 +1655,13 @@ class SparseArray(OpsMixin, PandasObject, ExtensionArray):
         non_nans = values[~mask]
         non_nan_idx = idx[~mask]

+        # Handle case where all values are fill_value or NaN
+        if len(non_nans) == 0:
+            # All values equal fill_value, return first valid index
+            _loc = self._first_fill_value_loc()
+            if _loc >= 0:
+                return _loc
+            return 0
+
         _candidate = non_nan_idx[func(non_nans)]
         candidate = index[_candidate]
```