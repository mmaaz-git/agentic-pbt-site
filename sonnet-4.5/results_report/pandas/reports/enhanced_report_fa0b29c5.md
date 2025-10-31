# Bug Report: pandas.core.arrays.sparse.SparseArray argmin/argmax Crash on All-Fill-Value Arrays

**Target**: `pandas.core.arrays.sparse.SparseArray.argmin` and `pandas.core.arrays.sparse.SparseArray.argmax`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

SparseArray.argmin() and SparseArray.argmax() crash with `ValueError: attempt to get argmin of an empty sequence` when all array values equal the fill value, violating the expected behavior that these methods should return the index of the first occurrence (0) like NumPy arrays do.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy as np
from pandas.core.arrays.sparse import SparseArray

@given(
    st.lists(
        st.integers(min_value=-100, max_value=100),
        min_size=2,
        max_size=50,
    )
)
def test_argmin_argmax_consistency(values):
    arr = np.array(values)
    sparse = SparseArray(arr)

    assert sparse.argmin() == np.argmin(arr)
    assert sparse.argmax() == np.argmax(arr)

if __name__ == "__main__":
    test_argmin_argmax_consistency()
```

<details>

<summary>
**Failing input**: `[0, 0]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/12/hypo.py", line 20, in <module>
    test_argmin_argmax_consistency()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/12/hypo.py", line 6, in test_argmin_argmax_consistency
    st.lists(

  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/12/hypo.py", line 16, in test_argmin_argmax_consistency
    assert sparse.argmin() == np.argmin(arr)
           ~~~~~~~~~~~~~^^
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
    values=[0, 0],
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
from pandas.core.arrays.sparse import SparseArray

# Create arrays where all values equal the fill_value (default is 0)
arr = np.array([0, 0])
sparse = SparseArray(arr)

print(f"Array: {arr}")
print(f"Fill value: {sparse.fill_value}")
print(f"Sparse values: {sparse.sp_values}")
print(f"Number of sparse points: {sparse.sp_index.npoints}")

# Show that numpy handles this case correctly
print(f"\nnp.argmin(arr): {np.argmin(arr)}")
print(f"np.argmax(arr): {np.argmax(arr)}")

# Try to call argmin on the sparse array (this should crash)
try:
    print(f"\nSparse.argmin(): {sparse.argmin()}")
except Exception as e:
    print(f"\nSparse.argmin() raised {type(e).__name__}: {e}")

# Try to call argmax on the sparse array (this should also crash)
try:
    print(f"\nSparse.argmax(): {sparse.argmax()}")
except Exception as e:
    print(f"Sparse.argmax() raised {type(e).__name__}: {e}")
```

<details>

<summary>
ValueError: attempt to get argmin/argmax of an empty sequence
</summary>
```
Array: [0 0]
Fill value: 0
Sparse values: []
Number of sparse points: 0

np.argmin(arr): 0
np.argmax(arr): 0

Sparse.argmin() raised ValueError: attempt to get argmin of an empty sequence
Sparse.argmax() raised ValueError: attempt to get argmax of an empty sequence

```
</details>

## Why This Is A Bug

This violates the expected behavior of argmin/argmax methods in multiple ways:

1. **NumPy Consistency Violation**: NumPy's argmin/argmax return index 0 when all values are equal. The NumPy documentation explicitly states: "In case of multiple occurrences of the minimum values, the indices corresponding to the first occurrence are returned." When all values are equal, they all qualify as both minimum and maximum, so index 0 should be returned.

2. **Pandas Series Consistency Violation**: pandas.Series.argmin() and argmax() follow NumPy's behavior and return 0 for arrays with all equal values. SparseArray should maintain consistency with other pandas array types.

3. **Implementation Flaw**: The crash occurs because when all values equal the fill_value, SparseArray stores no explicit values in its `sp_values` array (this is the sparse storage optimization). The `_argmin_argmax` method at line 1658 attempts to call `np.argmin/argmax` on the filtered `non_nans` array without checking if it's empty:
   ```python
   _candidate = non_nan_idx[func(non_nans)]  # Crashes when non_nans is empty
   ```

4. **API Contract Violation**: SparseArray is intended to be a drop-in replacement for regular arrays with sparse storage optimization. Users should not need to handle special cases when switching between dense and sparse representations.

## Relevant Context

The bug occurs in the `_argmin_argmax` method in `/pandas/core/arrays/sparse/array.py`. The method correctly handles cases where some non-fill values exist, and it has logic to compare the minimum/maximum sparse value with the fill_value. However, it fails to handle the edge case where there are no sparse values at all (when `sp_values` is empty).

The method already has access to `_first_fill_value_loc()` which returns the index of the first fill value in the array. When all values are fill values, this would correctly return 0, which is the expected result for both argmin and argmax in this case.

Documentation references:
- NumPy argmin: https://numpy.org/doc/stable/reference/generated/numpy.argmin.html
- NumPy argmax: https://numpy.org/doc/stable/reference/generated/numpy.argmax.html
- Pandas Series.argmin: https://pandas.pydata.org/docs/reference/api/pandas.Series.argmin.html

## Proposed Fix

```diff
--- a/pandas/core/arrays/sparse/array.py
+++ b/pandas/core/arrays/sparse/array.py
@@ -1648,6 +1648,14 @@ class SparseArray(OpsMixin, PandasObject, ExtensionArray):
     def _argmin_argmax(self, kind: Literal["argmin", "argmax"]) -> int:
         values = self._sparse_values
         index = self._sparse_index.indices
+
+        # Handle case where all values are fill_value (no sparse values stored)
+        if len(values) == 0:
+            # When all values equal fill_value, return the first index
+            _loc = self._first_fill_value_loc()
+            if _loc != -1:
+                return _loc
+            return 0  # Empty array or all NaN case
+
         mask = np.asarray(isna(values))
         func = np.argmax if kind == "argmax" else np.argmin
```