# Bug Report: pandas.core.array_algos.masked_reductions Object Dtype Axis Error

**Target**: `pandas.core.array_algos.masked_reductions`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The masked reduction functions (sum, prod, mean, var, std) crash when called on 2D object-dtype arrays with a non-None axis parameter. This is due to incorrect handling of object dtype arrays where the mask is applied via fancy indexing, flattening the array to 1D, before passing it to numpy functions that expect the original dimensionality.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st, settings
from pandas.core.array_algos.masked_reductions import sum as masked_sum
from pandas._libs import missing as libmissing


@given(
    rows=st.integers(min_value=2, max_value=5),
    cols=st.integers(min_value=2, max_value=5),
)
@settings(max_examples=100)
def test_masked_sum_object_dtype_with_axis(rows, cols):
    values_obj = np.arange(rows * cols).reshape(rows, cols).astype(object)
    mask = np.zeros((rows, cols), dtype=bool)
    mask[0, 0] = True

    result_axis1_obj = masked_sum(values_obj, mask, skipna=True, axis=1)
    result_axis1_float = masked_sum(values_obj.astype(float), mask, skipna=True, axis=1)

    if result_axis1_obj is not libmissing.NA and result_axis1_float is not libmissing.NA:
        if hasattr(result_axis1_obj, 'shape') and hasattr(result_axis1_float, 'shape'):
            assert result_axis1_obj.shape == result_axis1_float.shape
            for i in range(len(result_axis1_obj)):
                assert result_axis1_obj[i] == result_axis1_float[i]
```

**Failing input**: `rows=2, cols=2`

## Reproducing the Bug

```python
import numpy as np
from pandas.core.array_algos.masked_reductions import sum as masked_sum

values_obj = np.array([[1, 2], [3, 4]], dtype=object)
mask = np.array([[True, False], [False, False]], dtype=bool)

result = masked_sum(values_obj, mask, skipna=True, axis=1)
```

Output:
```
numpy.exceptions.AxisError: axis 1 is out of bounds for array of dimension 1
```

Expected: `array([2, 7], dtype=object)` (same as float dtype behavior)

## Why This Is A Bug

The masked reduction functions should behave consistently across different dtypes. When using float or other numeric dtypes, `masked_sum(arr, mask, axis=1)` correctly computes the sum along axis 1, respecting the mask. However, with object dtype, the function crashes because it applies the mask via fancy indexing (`values[~mask]`), which flattens the array to 1D, but then passes the original `axis` parameter to numpy functions that expect a 2D array.

This violates the documented API contract that these functions should work with any dtype that supports the operation, and it makes object dtype arrays behave differently from numeric arrays.

## Fix

```diff
--- a/pandas/core/array_algos/masked_reductions.py
+++ b/pandas/core/array_algos/masked_reductions.py
@@ -63,9 +63,14 @@ def _reductions(
             return libmissing.NA

         if values.dtype == np.dtype(object):
-            # object dtype does not support `where` without passing an initial
-            values = values[~mask]
-            return func(values, axis=axis, **kwargs)
+            if axis is None or values.ndim == 1:
+                # object dtype does not support `where` without passing an initial
+                values = values[~mask]
+                return func(values, axis=axis, **kwargs)
+            else:
+                # For multidimensional object arrays with axis, we need to use a different approach
+                # Fall back to the where-based approach with initial=0
+                return func(values, where=~mask, axis=axis, initial=0, **kwargs)
         return func(values, where=~mask, axis=axis, **kwargs)
```