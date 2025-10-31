# Bug Report: pandas.core.array_algos.masked_reductions min/max axis parameter

**Target**: `pandas.core.array_algos.masked_reductions.min` and `pandas.core.array_algos.masked_reductions.max`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `min()` and `max()` functions in `pandas.core.array_algos.masked_reductions` incorrectly handle the `axis` parameter for multi-dimensional arrays when `skipna=True` and there are no masked values. The functions flatten the input array via boolean indexing, causing the axis parameter to become invalid.

## Property-Based Test

```python
import numpy as np
import pandas.core.array_algos.masked_reductions as mr
from hypothesis import given, strategies as st

@given(
    st.integers(min_value=2, max_value=10),
    st.integers(min_value=2, max_value=10)
)
def test_min_2d_axis_0(rows, cols):
    values = np.random.rand(rows, cols)
    mask = np.zeros((rows, cols), dtype=bool)

    result = mr.min(values, mask, skipna=True, axis=0)
    expected = np.min(values, axis=0)

    np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-10)
```

**Failing input**: `rows=2, cols=2` with any values and all-False mask

## Reproducing the Bug

```python
import numpy as np
import pandas.core.array_algos.masked_reductions as mr

values = np.array([[0.548814, 0.715189],
                   [0.602763, 0.544883]])
mask = np.zeros((2, 2), dtype=bool)

result = mr.min(values, mask, skipna=True, axis=0)
expected = np.min(values, axis=0)

print(f"Result: {result}")
print(f"Expected: {expected}")

values2 = np.array([[0.548814, 0.715189],
                    [0.602763, 0.544883]])
mask2 = np.zeros((2, 2), dtype=bool)

result2 = mr.max(values2, mask2, skipna=True, axis=1)
```

**Output:**
```
Result: 0.544883
Expected: [0.548814 0.544883]
AxisError: axis 1 is out of bounds for array of dimension 1
```

## Why This Is A Bug

The functions are documented to accept an `axis` parameter and should behave like NumPy's `min`/`max` functions for multi-dimensional arrays. However, when `skipna=True`, the implementation uses `subset = values[~mask]`, which flattens the array through boolean indexing. This causes:

1. **For axis=0**: Returns a scalar instead of an array of minimums along axis 0
2. **For axis=1**: Raises `AxisError` because the flattened array has no axis 1

This violates the expected behavior documented in the docstring and breaks compatibility with NumPy's axis-based reduction semantics.

## Fix

```diff
--- a/pandas/core/array_algos/masked_reductions.py
+++ b/pandas/core/array_algos/masked_reductions.py
@@ -120,10 +120,22 @@ def _minmax(
             return libmissing.NA
         else:
             return func(values, axis=axis)
     else:
-        subset = values[~mask]
-        if subset.size:
-            return func(subset, axis=axis)
+        if axis is None:
+            subset = values[~mask]
+            if subset.size:
+                return func(subset)
+            else:
+                return libmissing.NA
+        elif mask.any():
+            subset = np.where(mask, np.nan, values)
+            if subset.size:
+                return np.nanmin(subset, axis=axis) if func is np.min else np.nanmax(subset, axis=axis)
+            else:
+                return libmissing.NA
         else:
-            # min/max with empty array raise in numpy, pandas returns NA
-            return libmissing.NA
+            if values.size:
+                return func(values, axis=axis)
+            else:
+                return libmissing.NA
```