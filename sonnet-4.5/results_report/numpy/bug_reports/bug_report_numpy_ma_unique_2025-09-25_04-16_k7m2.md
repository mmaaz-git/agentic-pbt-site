# Bug Report: numpy.ma.unique Multiple Masked Values

**Target**: `numpy.ma.unique`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`numpy.ma.unique()` violates its documented contract by returning multiple masked values instead of treating all masked values as a single element.

## Property-Based Test

```python
import numpy as np
import numpy.ma as ma
from hypothesis import given, strategies as st, settings
from hypothesis.extra import numpy as npst

@st.composite
def masked_arrays(draw, dtype=np.int64, max_dims=1, max_side=20):
    shape = draw(npst.array_shapes(max_dims=max_dims, max_side=max_side))
    data = draw(npst.arrays(dtype=dtype, shape=shape))
    mask = draw(npst.arrays(dtype=bool, shape=shape))
    return ma.array(data, mask=mask)

@given(masked_arrays())
@settings(max_examples=500)
def test_unique_treats_all_masked_as_one(arr):
    unique_vals = ma.unique(arr)
    masked_count = sum(1 for val in unique_vals if ma.is_masked(val))
    if ma.getmaskarray(arr).any():
        assert masked_count <= 1
```

**Failing input**: `masked_array(data=[--, 9223372036854775807, --], mask=[True, False, True])`

## Reproducing the Bug

```python
import numpy as np
import numpy.ma as ma

arr = ma.array([999, 9223372036854775807, 888], mask=[True, False, True])
unique_result = ma.unique(arr)

print(f"Input: {arr}")
print(f"Unique: {unique_result}")

masked_count = sum(1 for val in unique_result if ma.is_masked(val))
print(f"Number of masked values: {masked_count}")
print(f"Expected: At most 1")
print(f"Actual: {masked_count}")
```

## Why This Is A Bug

The docstring for `ma.unique()` explicitly states: "Masked values are considered the same element (masked)."

However, when an array contains multiple masked positions with different underlying data values, `unique()` treats them as distinct elements and returns multiple masked values in the result. This violates the documented semantics.

The root cause is that the implementation calls `np.unique()` on the raw data without accounting for the mask, then simply applies the mask to the result. Since `np.unique()` operates on underlying data values (999 and 888 in the example), it treats them as distinct, leading to multiple masked entries in the output.

## Fix

```diff
--- a/numpy/ma/extras.py
+++ b/numpy/ma/extras.py
@@ -172,6 +172,16 @@ def unique(ar1, return_index=False, return_inverse=False):
         fill_value=999999), array([0, 1, 4, 2]))
     """
+    ar1 = np.ma.asanyarray(ar1)
+    if np.ma.is_masked(ar1):
+        mask = np.ma.getmaskarray(ar1)
+        if mask.any():
+            data = np.ma.getdata(ar1).copy()
+            data[mask] = ar1.fill_value
+            ar1 = np.ma.array(data, mask=mask, fill_value=ar1.fill_value)
+
     output = np.unique(ar1,
                        return_index=return_index,
                        return_inverse=return_inverse)
```

The fix ensures all masked values have the same underlying data value (the fill_value) before calling `np.unique()`, so they are correctly treated as a single unique element.