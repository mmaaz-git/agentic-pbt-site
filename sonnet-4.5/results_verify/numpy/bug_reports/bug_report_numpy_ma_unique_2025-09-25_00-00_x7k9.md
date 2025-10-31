# Bug Report: numpy.ma.unique Multiple Masked Values

**Target**: `numpy.ma.unique`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`ma.unique()` violates its documented behavior by returning multiple masked elements instead of treating all masked values as a single element.

## Property-Based Test

```python
import numpy as np
import numpy.ma as ma
from hypothesis import given, strategies as st, assume
from hypothesis.extra import numpy as npst

@given(npst.arrays(dtype=npst.integer_dtypes(), shape=npst.array_shapes()),
       st.data())
def test_unique_treats_masked_as_equal(arr, data):
    assume(arr.size > 1)
    mask = data.draw(npst.arrays(dtype=np.bool_, shape=arr.shape))
    assume(np.sum(mask) >= 2)

    marr = ma.array(arr, mask=mask)

    unique_result = ma.unique(marr)

    masked_in_result = ma.getmaskarray(unique_result)
    assert np.sum(masked_in_result) <= 1
```

**Failing input**: `arr=array([32767, 32767, 32767], dtype=int16)`, `mask=array([True, False, True])`

## Reproducing the Bug

```python
import numpy as np
import numpy.ma as ma

arr = np.array([32767, 32767, 32767], dtype=np.int16)
mask = np.array([True, False, True])
marr = ma.array(arr, mask=mask)

unique_result = ma.unique(marr)
print('Result:', unique_result)
print('Number of masked values:', np.sum(ma.getmaskarray(unique_result)))
```

Expected output: 1 masked value (doc states "Masked values are considered the same element")
Actual output: 2 masked values

## Why This Is A Bug

The documentation explicitly states: "Masked values are considered the same element (masked)." This means all masked values, regardless of their underlying data values, should be treated as identical and collapse to a single masked element in the unique output.

However, the current implementation simply calls `np.unique()` on the masked array and converts the result to a MaskedArray view. Since `np.unique()` doesn't understand masks, it operates on the underlying data separately from the mask, leading to incorrect results when masked elements share the same underlying value with unmasked elements.

## Fix

The function needs to handle masked values explicitly before calling `np.unique()`. One approach:

1. Extract unmasked values
2. Find unique unmasked values
3. If any values were masked, append a single masked element

```diff
--- a/numpy/ma/extras.py
+++ b/numpy/ma/extras.py
@@ -397,11 +397,24 @@ def unique(ar1, return_index=False, return_inverse=False):
     (masked_array(data=[1, 2, 3, --],
                 mask=[False, False, False,  True],
         fill_value=999999), array([0, 1, 4, 2]), array([0, 1, 3, 1, 2]))
     """
-    output = np.unique(ar1,
-                       return_index=return_index,
-                       return_inverse=return_inverse)
-    if isinstance(output, tuple):
-        output = list(output)
-        output[0] = output[0].view(MaskedArray)
-        output = tuple(output)
-    else:
-        output = output.view(MaskedArray)
-    return output
+    # Separate masked and unmasked values
+    mask = getmaskarray(ar1)
+    has_masked = np.any(mask)
+
+    if has_masked:
+        # Get unique of unmasked values
+        unmasked_data = ar1.compressed()
+        unique_unmasked = np.unique(unmasked_data,
+                                   return_index=return_index,
+                                   return_inverse=return_inverse)
+        # Add single masked element
+        if isinstance(unique_unmasked, tuple):
+            unique_vals = unique_unmasked[0]
+            result_data = np.append(unique_vals, ar1.fill_value)
+            result_mask = np.append(np.zeros(len(unique_vals), dtype=bool), True)
+            result = ma.array(result_data, mask=result_mask)
+            # Adjust indices and inverse if requested
+            # ... (complex logic needed for return_index and return_inverse)
+            return (result,) + unique_unmasked[1:]
+        else:
+            result_data = np.append(unique_unmasked, ar1.fill_value)
+            result_mask = np.append(np.zeros(len(unique_unmasked), dtype=bool), True)
+            return ma.array(result_data, mask=result_mask)
+    else:
+        # No masked values, use original implementation
+        output = np.unique(ar1,
+                          return_index=return_index,
+                          return_inverse=return_inverse)
+        if isinstance(output, tuple):
+            output = list(output)
+            output[0] = output[0].view(MaskedArray)
+            output = tuple(output)
+        else:
+            output = output.view(MaskedArray)
+        return output
```