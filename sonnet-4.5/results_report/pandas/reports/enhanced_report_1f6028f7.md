# Bug Report: pandas.core.arrays.sparse.SparseArray Ignores skipna=False in max/min Operations

**Target**: `pandas.core.arrays.sparse.SparseArray._min_max`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

SparseArray.max() and SparseArray.min() incorrectly return numeric values instead of NaN when called with skipna=False on arrays containing NaN values and a non-null fill_value, violating NumPy/pandas conventions and causing silent data corruption.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy as np
from pandas.arrays import SparseArray

@given(st.lists(
    st.one_of(
        st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10),
        st.just(np.nan)
    ),
    min_size=1,
    max_size=50
))
def test_max_min_skipna_false(values):
    arr = SparseArray(values, fill_value=0.0)

    if any(np.isnan(v) for v in values):
        # Test max
        sparse_max = arr.max(skipna=False)
        dense_max = np.max(arr.to_dense())

        assert np.isnan(sparse_max) and np.isnan(dense_max), \
            f"When array contains NaN and skipna=False, max should return NaN. Got {sparse_max} for values {values}"

        # Test min
        sparse_min = arr.min(skipna=False)
        dense_min = np.min(arr.to_dense())

        assert np.isnan(sparse_min) and np.isnan(dense_min), \
            f"When array contains NaN and skipna=False, min should return NaN. Got {sparse_min} for values {values}"

# Run the test
if __name__ == "__main__":
    test_max_min_skipna_false()
```

<details>

<summary>
**Failing input**: `[nan, 0.0]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/43/hypo.py", line 33, in <module>
    test_max_min_skipna_false()
    ~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/43/hypo.py", line 6, in test_max_min_skipna_false
    st.one_of(

  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/43/hypo.py", line 21, in test_max_min_skipna_false
    assert np.isnan(sparse_max) and np.isnan(dense_max), \
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: When array contains NaN and skipna=False, max should return NaN. Got 0.0 for values [nan, 0.0]
Falsifying example: test_max_min_skipna_false(
    values=[nan, 0.0],
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/worker_/43/hypo.py:18
        /home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/sparse/array.py:620
        /home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/sparse/array.py:634
        /home/npc/miniconda/lib/python3.13/site-packages/pandas/core/dtypes/missing.py:202
        /home/npc/miniconda/lib/python3.13/site-packages/pandas/core/dtypes/missing.py:204
        /home/npc/miniconda/lib/python3.13/site-packages/pandas/core/dtypes/missing.py:293
```
</details>

## Reproducing the Bug

```python
import numpy as np
from pandas.arrays import SparseArray

# Create a SparseArray with NaN values and a non-null fill_value
arr = SparseArray([1.0, np.nan, 0.0], fill_value=0.0)

# Test max with skipna=False
max_result = arr.max(skipna=False)
print(f"SparseArray.max(skipna=False): {max_result}")
print(f"Expected: nan")
print()

# Test min with skipna=False
min_result = arr.min(skipna=False)
print(f"SparseArray.min(skipna=False): {min_result}")
print(f"Expected: nan")
print()

# Compare with NumPy behavior
numpy_array = np.array([1.0, np.nan, 0.0])
numpy_max = np.max(numpy_array)
numpy_min = np.min(numpy_array)
print(f"NumPy max: {numpy_max}")
print(f"NumPy min: {numpy_min}")
print()

# Compare with dense pandas Series
import pandas as pd
series = pd.Series([1.0, np.nan, 0.0])
series_max = series.max(skipna=False)
series_min = series.min(skipna=False)
print(f"Series.max(skipna=False): {series_max}")
print(f"Series.min(skipna=False): {series_min}")
```

<details>

<summary>
Incorrect behavior - SparseArray returns numeric values instead of NaN
</summary>
```
SparseArray.max(skipna=False): 1.0
Expected: nan

SparseArray.min(skipna=False): 0.0
Expected: nan

NumPy max: nan
NumPy min: nan

Series.max(skipna=False): nan
Series.min(skipna=False): nan
```
</details>

## Why This Is A Bug

This violates fundamental NumPy and pandas behavior conventions. The `skipna` parameter explicitly controls whether NA/NaN values should be ignored during aggregation operations. When `skipna=False`, the universally established convention across NumPy and pandas is that NaN values propagate through calculations, resulting in NaN output.

The bug occurs in `pandas/core/arrays/sparse/array.py` at line 1626 in the `_min_max` method, which unconditionally uses `self._valid_sp_values`. This property (defined at line 674) always filters out NaN values using `notna(sp_vals)`, regardless of the `skipna` parameter value. When the array has a non-null fill_value (line 1627 check for `has_nonnull_fill_vals`), the method proceeds to compute and return numeric min/max values even though NaN values are present and `skipna=False` was specified.

This causes silent data corruption where users receive numeric results when they should receive NaN, potentially leading to incorrect statistical conclusions. The behavior is inconsistent with both NumPy (`np.max([1.0, np.nan, 0.0])` returns NaN) and dense pandas arrays (`pd.Series([1.0, np.nan, 0.0]).max(skipna=False)` returns NaN), breaking the principle that SparseArray should be a drop-in replacement for dense arrays.

## Relevant Context

The SparseArray documentation states that the `skipna` parameter controls "Whether to ignore NA values" (line 1604 in array.py). This clearly implies that when `skipna=False`, NA values should NOT be ignored, meaning they should affect the computation result by propagating as NaN.

The bug specifically manifests when:
1. The SparseArray has a non-null fill_value (e.g., 0.0)
2. The array contains NaN values in its sparse values
3. The `skipna=False` parameter is explicitly passed

Key code locations:
- `pandas/core/arrays/sparse/array.py:1626` - The problematic line that always uses filtered values
- `pandas/core/arrays/sparse/array.py:674-676` - The `_valid_sp_values` property that unconditionally filters NaN
- `pandas/core/arrays/sparse/array.py:1633-1635` - The code path that returns numeric values when it should return NaN

## Proposed Fix

```diff
--- a/pandas/core/arrays/sparse/array.py
+++ b/pandas/core/arrays/sparse/array.py
@@ -1623,7 +1623,14 @@ class SparseArray(OpsMixin, PandasObject, ExtensionArray):
         -------
         scalar
         """
-        valid_vals = self._valid_sp_values
+        # When skipna=True, filter out NaN values. When skipna=False, keep them.
+        if skipna:
+            valid_vals = self._valid_sp_values
+        else:
+            # Check if any NaN values exist in sp_values
+            if len(self.sp_values) > 0 and isna(self.sp_values).any():
+                return na_value_for_dtype(self.dtype.subtype, compat=False)
+            valid_vals = self.sp_values
         has_nonnull_fill_vals = not self._null_fill_value and self.sp_index.ngaps > 0

         if len(valid_vals) > 0:
```