# Bug Report: pandas.core.arrays.sparse.SparseArray.sum() Fails to Propagate NaN When skipna=False

**Target**: `pandas.core.arrays.sparse.SparseArray.sum()`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `SparseArray.sum(skipna=False)` method incorrectly returns a numeric value instead of NaN when the sparse array contains NaN values in its `sp_values`, violating the fundamental pandas contract that reduction operations with `skipna=False` should propagate NaN values.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings, example
import numpy as np
from pandas.core.arrays.sparse import SparseArray

@given(st.lists(
    st.floats(allow_nan=True, allow_infinity=False),
    min_size=1, max_size=100
))
@example(data=[np.nan])  # The minimal failing case
@settings(max_examples=100)
def test_sum_matches_dense_skipna_false(data):
    """Test that SparseArray.sum(skipna=False) properly handles NaN values."""
    arr = SparseArray(data, fill_value=0.0)

    sparse_sum = arr.sum(skipna=False)
    dense_sum = arr.to_dense().sum()

    # Both should be NaN if there's any NaN in the data
    if np.isnan(dense_sum):
        assert np.isnan(sparse_sum), f"Expected NaN but got {sparse_sum} for data={data}"
    else:
        # Check if they're close (accounting for floating point precision)
        assert np.isclose(sparse_sum, dense_sum, rtol=1e-10, equal_nan=True), \
            f"Expected {dense_sum} but got {sparse_sum} for data={data}"

if __name__ == "__main__":
    # Run the test
    test_sum_matches_dense_skipna_false()
```

<details>

<summary>
**Failing input**: `[nan]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/55/hypo.py", line 28, in <module>
    test_sum_matches_dense_skipna_false()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/55/hypo.py", line 6, in test_sum_matches_dense_skipna_false
    st.floats(allow_nan=True, allow_infinity=False),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2062, in wrapped_test
    _raise_to_user(errors, state.settings, [], " in explicit examples")
    ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1613, in _raise_to_user
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/55/hypo.py", line 20, in test_sum_matches_dense_skipna_false
    assert np.isnan(sparse_sum), f"Expected NaN but got {sparse_sum} for data={data}"
           ~~~~~~~~^^^^^^^^^^^^
AssertionError: Expected NaN but got 0.0 for data=[nan]
Falsifying explicit example: test_sum_matches_dense_skipna_false(
    data=[nan],
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
import pandas as pd
from pandas.core.arrays.sparse import SparseArray

# Test case showing the bug
arr = SparseArray([np.nan], fill_value=0.0)

print("Expected behavior (pandas Series):")
s = pd.Series([np.nan])
print(f"  pd.Series([np.nan]).sum(skipna=False) = {s.sum(skipna=False)}")
print(f"  pd.Series([np.nan]).sum(skipna=True) = {s.sum(skipna=True)}")

print("\nActual behavior (SparseArray):")
print(f"  SparseArray([np.nan]).sum(skipna=False) = {arr.sum(skipna=False)}")
print(f"  SparseArray([np.nan]).sum(skipna=True) = {arr.sum(skipna=True)}")

print("\n--- More examples ---")
print("\nExample with mix of NaN and regular values:")
arr2 = SparseArray([1.0, np.nan, 2.0], fill_value=0.0)
s2 = pd.Series([1.0, np.nan, 2.0])

print(f"Data: [1.0, np.nan, 2.0]")
print(f"  pd.Series.sum(skipna=False) = {s2.sum(skipna=False)}")
print(f"  SparseArray.sum(skipna=False) = {arr2.sum(skipna=False)}")
print(f"  pd.Series.sum(skipna=True) = {s2.sum(skipna=True)}")
print(f"  SparseArray.sum(skipna=True) = {arr2.sum(skipna=True)}")

print("\n--- Internal details ---")
print(f"\nFor SparseArray([np.nan]):")
print(f"  sp_values: {arr.sp_values}")
print(f"  fill_value: {arr.fill_value}")
print(f"  _valid_sp_values: {arr._valid_sp_values}")
print(f"  Length of sp_values: {len(arr.sp_values)}")
print(f"  Length of _valid_sp_values: {len(arr._valid_sp_values)}")
```

<details>

<summary>
SparseArray.sum() returns 0.0 instead of NaN when skipna=False
</summary>
```
Expected behavior (pandas Series):
  pd.Series([np.nan]).sum(skipna=False) = nan
  pd.Series([np.nan]).sum(skipna=True) = 0.0

Actual behavior (SparseArray):
  SparseArray([np.nan]).sum(skipna=False) = 0.0
  SparseArray([np.nan]).sum(skipna=True) = 0.0

--- More examples ---

Example with mix of NaN and regular values:
Data: [1.0, np.nan, 2.0]
  pd.Series.sum(skipna=False) = nan
  SparseArray.sum(skipna=False) = 3.0
  pd.Series.sum(skipna=True) = 3.0
  SparseArray.sum(skipna=True) = 3.0

--- Internal details ---

For SparseArray([np.nan]):
  sp_values: [nan]
  fill_value: 0.0
  _valid_sp_values: []
  Length of sp_values: 1
  Length of _valid_sp_values: 0
```
</details>

## Why This Is A Bug

This bug violates the fundamental pandas behavior for reduction operations with `skipna=False`. According to pandas documentation and consistent behavior across all array types, when `skipna=False` is specified, any NaN values in the data should cause the reduction operation to return NaN. This is critical for data integrity in scientific computing and data analysis where missing values must be properly propagated.

The bug occurs because:
1. The `_valid_sp_values` property (line 674-677 in array.py) filters out NaN values from `sp_values` using `notna(sp_vals)`
2. The `sum()` method (lines 1508-1524) only checks for NaN in "gaps" (positions with fill_value) via the condition `has_na = self.sp_index.ngaps > 0 and not self._null_fill_value`
3. It fails to detect when NaN values exist in the explicitly stored sparse values themselves
4. When `skipna=False`, the method should return NaN if ANY values (stored or fill) are NaN, but it only checks fill values

This leads to incorrect results where `SparseArray([np.nan]).sum(skipna=False)` returns `0.0` instead of `nan`, and `SparseArray([1.0, np.nan, 2.0]).sum(skipna=False)` returns `3.0` instead of `nan`.

## Relevant Context

The issue is located in `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/core/arrays/sparse/array.py` at lines 1508-1524 in the `sum()` method.

Key methods involved:
- `_valid_sp_values` property (lines 674-677): Filters NaN from sparse values
- `sum()` method (lines 1508-1524): Performs the summation but only checks for NaN in gaps

This bug affects all pandas users working with sparse data structures where NaN values may appear in the data. The SparseArray is commonly used for memory-efficient storage of data with many repeated values (like zeros), and proper NaN handling is crucial for data analysis pipelines.

## Proposed Fix

```diff
--- a/pandas/core/arrays/sparse/array.py
+++ b/pandas/core/arrays/sparse/array.py
@@ -1508,11 +1508,17 @@ class SparseArray(OpsMixin, PandasObject, ExtensionArray):
         nv.validate_sum(args, kwargs)
         valid_vals = self._valid_sp_values
         sp_sum = valid_vals.sum()
         has_na = self.sp_index.ngaps > 0 and not self._null_fill_value

+        # Check if sp_values contains any NaN values
+        if not skipna and len(valid_vals) < len(self.sp_values):
+            # Some sp_values were filtered out as NaN in _valid_sp_values
+            return na_value_for_dtype(self.dtype.subtype, compat=False)
+
         if has_na and not skipna:
             return na_value_for_dtype(self.dtype.subtype, compat=False)

         if self._null_fill_value:
             if check_below_min_count(valid_vals.shape, None, min_count):
                 return na_value_for_dtype(self.dtype.subtype, compat=False)
```