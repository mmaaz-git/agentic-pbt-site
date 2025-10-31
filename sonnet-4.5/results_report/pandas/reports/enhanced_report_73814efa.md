# Bug Report: pandas.core.arrays.sparse.SparseArray.astype() Silently Corrupts Data When Changing fill_value

**Target**: `pandas.core.arrays.sparse.SparseArray.astype()`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

SparseArray.astype() silently replaces actual data values with the new fill_value when converting to a SparseDtype with a different fill_value, causing data corruption without warning.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy as np
import pandas as pd
from pandas.core.arrays.sparse import SparseArray

@given(
    data=st.lists(st.integers(min_value=-1000, max_value=1000), min_size=1, max_size=100),
    fill_value1=st.integers(min_value=-1000, max_value=1000),
    fill_value2=st.integers(min_value=-1000, max_value=1000)
)
def test_sparse_array_astype_preserves_values(data, fill_value1, fill_value2):
    arr = np.array(data)
    sparse = SparseArray(arr, fill_value=fill_value1)

    dtype = pd.SparseDtype(np.float64, fill_value2)
    sparse_casted = sparse.astype(dtype)

    np.testing.assert_allclose(sparse_casted.to_dense(), arr.astype(np.float64))

# Run the test to find a failing case
if __name__ == "__main__":
    # This will run the test and report any failures
    test_sparse_array_astype_preserves_values()
```

<details>

<summary>
**Failing input**: `data=[0], fill_value1=0, fill_value2=1`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/13/hypo.py", line 23, in <module>
    test_sparse_array_astype_preserves_values()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/13/hypo.py", line 7, in test_sparse_array_astype_preserves_values
    data=st.lists(st.integers(min_value=-1000, max_value=1000), min_size=1, max_size=100),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/13/hypo.py", line 18, in test_sparse_array_astype_preserves_values
    np.testing.assert_allclose(sparse_casted.to_dense(), arr.astype(np.float64))
    ~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/testing/_private/utils.py", line 1708, in assert_allclose
    assert_array_compare(compare, actual, desired, err_msg=str(err_msg),
    ~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                         verbose=verbose, header=header, equal_nan=equal_nan,
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                         strict=strict)
                         ^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/testing/_private/utils.py", line 916, in assert_array_compare
    raise AssertionError(msg)
AssertionError:
Not equal to tolerance rtol=1e-07, atol=0

Mismatched elements: 1 / 1 (100%)
Max absolute difference among violations: 1.
Max relative difference among violations: inf
 ACTUAL: array([1.])
 DESIRED: array([0.])
Falsifying example: test_sparse_array_astype_preserves_values(
    data=[0],
    fill_value1=0,
    fill_value2=1,
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/arrayprint.py:600
        /home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/arrayprint.py:964
        /home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/arrayprint.py:1016
        /home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/arrayprint.py:1021
        /home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/fromnumeric.py:3048
        (and 5 more with settings.verbosity >= verbose)
```
</details>

## Reproducing the Bug

```python
import numpy as np
import pandas as pd
from pandas.core.arrays.sparse import SparseArray

# Minimal failing case from the bug report
data = [0]
sparse = SparseArray(data, fill_value=0)
print("Original SparseArray:")
print("  Values:", sparse.to_dense())
print("  Fill value:", sparse.fill_value)
print("  sp_values:", sparse.sp_values)
print("  sp_index.indices:", sparse.sp_index.indices)
print()

# Cast to float64 with a different fill_value
dtype = pd.SparseDtype(np.float64, fill_value=1)
casted = sparse.astype(dtype)
print("After astype(SparseDtype(float64, fill_value=1)):")
print("  Values:", casted.to_dense())
print("  Fill value:", casted.fill_value)
print("  sp_values:", casted.sp_values)
print("  sp_index.indices:", casted.sp_index.indices)
print()

print("Expected: [0.]")
print("Actual:  ", casted.to_dense())
print()

# Verify the fundamental invariant is violated
print("Testing fundamental invariant:")
print("  sparse.astype(dtype).to_dense() =", casted.to_dense())
print("  sparse.to_dense().astype(np.float64) =", sparse.to_dense().astype(np.float64))
print("  These should be equal but they are not!")
```

<details>

<summary>
Data corruption confirmed: value 0 incorrectly becomes 1
</summary>
```
Original SparseArray:
  Values: [0]
  Fill value: 0
  sp_values: []
  sp_index.indices: []

After astype(SparseDtype(float64, fill_value=1)):
  Values: [1.]
  Fill value: 1
  sp_values: []
  sp_index.indices: []

Expected: [0.]
Actual:   [1.]

Testing fundamental invariant:
  sparse.astype(dtype).to_dense() = [1.]
  sparse.to_dense().astype(np.float64) = [0.]
  These should be equal but they are not!
```
</details>

## Why This Is A Bug

This bug violates the fundamental contract of `astype()` which must preserve array values while only changing their type representation. The bug causes **silent data corruption** where user values are replaced without any warning or error.

The issue occurs specifically when:
1. A SparseArray contains values equal to its fill_value (resulting in empty `sp_values` and `sp_index.indices`)
2. `astype()` is called with a SparseDtype that has a different fill_value
3. The original values that equaled the old fill_value are incorrectly replaced with the new fill_value

The bug contradicts pandas' own documentation. The docstring at lines 1287-1292 of `array.py` shows an example where `[0, 0, 1, 2]` with `fill_value=0` correctly becomes `[0.0, 0.0, 1.0, 2.0]` with `fill_value=0.0`, preserving all zeros. However, when ALL values equal the fill_value, they get incorrectly replaced.

This violates the mathematical invariant: `sparse.astype(dtype).to_dense()` should equal `sparse.to_dense().astype(dtype)`.

## Relevant Context

The root cause lies in how SparseArrays store data internally:
- **sp_values**: stores only non-fill values
- **sp_index.indices**: stores indices where non-fill values appear
- When ALL values equal fill_value, both arrays are empty

The `__array__` method (lines 593-594 in array.py) reconstructs the dense array by:
1. Creating an array filled with the current fill_value
2. Setting values at sp_index.indices from sp_values

When `astype()` is called with a new fill_value but sp_values is empty, the reconstructed array uses the NEW fill_value everywhere, losing the original data.

Pattern analysis shows:
- **All values = fill_value**: Complete data loss (e.g., [5,5,5] → [10,10,10])
- **Some values = fill_value**: Partial data loss (e.g., [5,6,5] → [10,6,10])
- **No values = fill_value**: Works correctly (e.g., [6,7,8] → [6,7,8])

Documentation: https://pandas.pydata.org/docs/reference/api/pandas.arrays.SparseArray.astype.html
Source code: `/pandas/core/arrays/sparse/array.py` lines 1240-1314

## Proposed Fix

The astype method needs to detect when fill_value changes and explicitly preserve values that were previously stored implicitly as fill values:

```diff
--- a/pandas/core/arrays/sparse/array.py
+++ b/pandas/core/arrays/sparse/array.py
@@ -1305,6 +1305,14 @@ class SparseArray(OpsMixin, PandasObject, ExtensionArray):
             return astype_array(values, dtype=future_dtype, copy=False)

         dtype = self.dtype.update_dtype(dtype)
+
+        # If fill_value is changing, we need to convert to dense first
+        # to preserve actual values, then convert back to sparse
+        if not self._fill_value_matches(dtype.fill_value):
+            dense_values = np.asarray(self)
+            dense_values = ensure_wrapped_if_datetimelike(dense_values)
+            dense_casted = astype_array(dense_values, dtype.subtype, copy=False)
+            return type(self)(dense_casted, fill_value=dtype.fill_value)
+
         subtype = pandas_dtype(dtype._subtype_with_str)
         subtype = cast(np.dtype, subtype)  # ensured by update_dtype
         values = ensure_wrapped_if_datetimelike(self.sp_values)
```