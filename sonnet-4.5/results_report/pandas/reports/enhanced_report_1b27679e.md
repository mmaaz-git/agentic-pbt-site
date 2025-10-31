# Bug Report: pandas SparseArray argmin/argmax Crash on Arrays with Only Fill Values

**Target**: `pandas.core.arrays.sparse.SparseArray.argmin()` and `SparseArray.argmax()`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

SparseArray's `argmin()` and `argmax()` methods crash with a ValueError when all elements in the array equal the fill_value, resulting in zero sparse values, whereas NumPy arrays with identical values correctly return index 0.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from pandas.core.arrays.sparse import SparseArray

@given(
    st.lists(st.integers(min_value=-100, max_value=100), min_size=1, max_size=50),
    st.integers(min_value=-100, max_value=100)
)
@settings(max_examples=300)
def test_argmin_argmax_consistency(data, fill_value):
    arr = SparseArray(data, fill_value=fill_value)
    dense = arr.to_dense()

    if len(arr) > 0:
        assert arr[arr.argmin()] == dense[dense.argmin()]
        assert arr[arr.argmax()] == dense[dense.argmax()]

# Run the test
if __name__ == "__main__":
    test_argmin_argmax_consistency()
```

<details>

<summary>
**Failing input**: `data=[0], fill_value=0`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/33/hypo.py", line 19, in <module>
    test_argmin_argmax_consistency()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/33/hypo.py", line 5, in test_argmin_argmax_consistency
    st.lists(st.integers(min_value=-100, max_value=100), min_size=1, max_size=50),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/33/hypo.py", line 14, in test_argmin_argmax_consistency
    assert arr[arr.argmin()] == dense[dense.argmin()]
               ~~~~~~~~~~^^
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
Falsifying example: test_argmin_argmax_consistency(
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

# Case 1: Single element array where element equals fill_value
print("Case 1: SparseArray([0], fill_value=0)")
arr = SparseArray([0], fill_value=0)
dense = arr.to_dense()

print(f"SparseArray: {arr}")
print(f"Dense array: {dense}")
print(f"npoints (sparse values): {arr.npoints}")
print(f"Length: {len(arr)}")

try:
    result = arr.argmin()
    print(f"argmin() result: {result}")
except Exception as e:
    print(f"argmin() raised: {type(e).__name__}: {e}")

try:
    result = arr.argmax()
    print(f"argmax() result: {result}")
except Exception as e:
    print(f"argmax() raised: {type(e).__name__}: {e}")

print("\nFor comparison, NumPy on the same data:")
np_arr = np.array([0])
print(f"NumPy array: {np_arr}")
print(f"np.argmin(): {np_arr.argmin()}")
print(f"np.argmax(): {np_arr.argmax()}")

print("\n" + "="*50 + "\n")

# Case 2: Multiple elements, all equal to fill_value
print("Case 2: SparseArray([5, 5, 5, 5], fill_value=5)")
arr2 = SparseArray([5, 5, 5, 5], fill_value=5)
dense2 = arr2.to_dense()

print(f"SparseArray: {arr2}")
print(f"Dense array: {dense2}")
print(f"npoints (sparse values): {arr2.npoints}")
print(f"Length: {len(arr2)}")

try:
    result = arr2.argmin()
    print(f"argmin() result: {result}")
except Exception as e:
    print(f"argmin() raised: {type(e).__name__}: {e}")

try:
    result = arr2.argmax()
    print(f"argmax() result: {result}")
except Exception as e:
    print(f"argmax() raised: {type(e).__name__}: {e}")

print("\nFor comparison, NumPy on the same data:")
np_arr2 = np.array([5, 5, 5, 5])
print(f"NumPy array: {np_arr2}")
print(f"np.argmin(): {np_arr2.argmin()}")
print(f"np.argmax(): {np_arr2.argmax()}")
```

<details>

<summary>
ValueError: attempt to get argmin of an empty sequence
</summary>
```
Case 1: SparseArray([0], fill_value=0)
SparseArray: [0]
Fill: 0
IntIndex
Indices: array([], dtype=int32)

Dense array: [0]
npoints (sparse values): 0
Length: 1
argmin() raised: ValueError: attempt to get argmin of an empty sequence
argmax() raised: ValueError: attempt to get argmax of an empty sequence

For comparison, NumPy on the same data:
NumPy array: [0]
np.argmin(): 0
np.argmax(): 0

==================================================

Case 2: SparseArray([5, 5, 5, 5], fill_value=5)
SparseArray: [5, 5, 5, 5]
Fill: 5
IntIndex
Indices: array([], dtype=int32)

Dense array: [5 5 5 5]
npoints (sparse values): 0
Length: 4
argmin() raised: ValueError: attempt to get argmin of an empty sequence
argmax() raised: ValueError: attempt to get argmax of an empty sequence

For comparison, NumPy on the same data:
NumPy array: [5 5 5 5]
np.argmin(): 0
np.argmax(): 0
```
</details>

## Why This Is A Bug

This violates expected behavior for several reasons:

1. **NumPy Compatibility**: NumPy arrays with all identical values return index 0 for both `argmin()` and `argmax()`. This is the standard behavior that users expect from array-like objects. For example, `np.array([5, 5, 5, 5]).argmin()` returns 0, not an error.

2. **ExtensionArray Contract**: SparseArray inherits from ExtensionArray, whose documentation states that argmin/argmax should "Return the index of minimum/maximum value. In case of multiple occurrences, the first occurrence is returned." There's no mention of raising errors for arrays with identical values.

3. **Misleading Error Message**: The error "attempt to get argmin of an empty sequence" is incorrect - the array is not empty. It has valid data (length > 0), but the internal sparse representation has no sparse values because everything equals the fill_value.

4. **Inconsistency with Dense Arrays**: When converted to dense via `to_dense()`, the same data works correctly with argmin/argmax. SparseArray should be a memory-efficient alternative to dense arrays, not one with different semantics.

5. **pandas.Series Behavior**: A pandas Series with identical values also returns 0 for argmin/argmax, establishing consistent behavior across pandas data structures.

## Relevant Context

The bug occurs in the `_argmin_argmax` method at `/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/sparse/array.py:1658`. When a SparseArray has all values equal to the fill_value:

- The `_sparse_values` attribute contains no values (empty array)
- The method filters out NaN values, resulting in `non_nans` being empty
- Calling `np.argmin()` or `np.argmax()` on this empty array raises a ValueError

The implementation already has logic to handle fill_value comparisons (lines 1661-1672), but this code is never reached because the error occurs earlier when trying to process the empty sparse values array.

Documentation references:
- [pandas.core.arrays.ExtensionArray](https://pandas.pydata.org/docs/reference/api/pandas.api.extensions.ExtensionArray.html)
- [pandas.arrays.SparseArray](https://pandas.pydata.org/docs/reference/api/pandas.arrays.SparseArray.html)

## Proposed Fix

```diff
--- a/pandas/core/arrays/sparse/array.py
+++ b/pandas/core/arrays/sparse/array.py
@@ -1655,6 +1655,11 @@ class SparseArray(OpsMixin, PandasObject, ExtensionArray):
         non_nans = values[~mask]
         non_nan_idx = idx[~mask]

+        # Handle case where all values are fill_value (no sparse values)
+        if len(non_nans) == 0:
+            # Return first fill value location, or 0 if array is all fill values
+            return 0 if len(self) > 0 else -1
+
         _candidate = non_nan_idx[func(non_nans)]
         candidate = index[_candidate]
```