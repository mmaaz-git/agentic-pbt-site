# Bug Report: numpy.ma.intersect1d Multiple Masked Values

**Target**: `numpy.ma.intersect1d`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`ma.intersect1d()` violates its documented behavior by returning multiple masked elements instead of treating all masked values as equal.

## Property-Based Test

```python
import numpy as np
import numpy.ma as ma
from hypothesis import given, strategies as st, assume
from hypothesis.extra import numpy as npst

@given(npst.arrays(dtype=npst.integer_dtypes(), shape=st.tuples(st.integers(2, 20))),
       npst.arrays(dtype=npst.integer_dtypes(), shape=st.tuples(st.integers(2, 20))),
       st.data())
def test_intersect1d_masked_handling(ar1, ar2, data):
    assume(ar1.size > 1 and ar2.size > 1)

    mask1 = data.draw(npst.arrays(dtype=np.bool_, shape=ar1.shape))
    mask2 = data.draw(npst.arrays(dtype=np.bool_, shape=ar2.shape))

    assume(np.sum(mask1) >= 1 and np.sum(mask2) >= 1)

    mar1 = ma.array(ar1, mask=mask1)
    mar2 = ma.array(ar2, mask=mask2)

    intersection = ma.intersect1d(mar1, mar2)

    masked_in_result = ma.getmaskarray(intersection)
    if masked_in_result is not ma.nomask:
        assert np.sum(masked_in_result) <= 1
```

**Failing input**: `ar1=array([0, 0], dtype=int16)` with `mask=[True, False]`, `ar2=array([0, 127, 0], dtype=int8)` with `mask=[True, False, True]`

## Reproducing the Bug

```python
import numpy as np
import numpy.ma as ma

ar1 = np.array([0, 0], dtype=np.int16)
mask1 = np.array([True, False])
mar1 = ma.array(ar1, mask=mask1)

ar2 = np.array([0, 127, 0], dtype=np.int8)
mask2 = np.array([True, False, True])
mar2 = ma.array(ar2, mask=mask2)

intersection = ma.intersect1d(mar1, mar2)
print('Result:', intersection)
print('Number of masked values:', np.sum(ma.getmaskarray(intersection)))
```

Expected: At most 1 masked value (doc says "Masked values are considered equal one to the other")
Actual: 2 masked values

## Why This Is A Bug

The documentation states: "Masked values are considered equal one to the other." This means all masked values should be treated as identical, regardless of their underlying data values. The intersection of two sets of masked values should yield at most one masked element.

The root cause is in the implementation which calls `unique()` on both arrays (which itself has a bug returning multiple masked values), then concatenates and filters. Since `unique()` can return multiple masked elements, those propagate through to the final result.

## Fix

This bug is a consequence of the bug in `ma.unique()`. Once `unique()` is fixed to properly collapse all masked values into a single element, `intersect1d()` will work correctly. Alternatively, handle masked values explicitly:

```diff
--- a/numpy/ma/extras.py
+++ b/numpy/ma/extras.py
@@ -450,8 +450,23 @@ def intersect1d(ar1, ar2, assume_unique=False):
     >>> np.ma.intersect1d(x, y)
     masked_array(data=[1, 3, --],
                  mask=[False, False,  True],
            fill_value=999999)
     """
+    # Handle masked values explicitly
+    has_masked1 = np.any(getmaskarray(ar1))
+    has_masked2 = np.any(getmaskarray(ar2))
+
+    # Get intersection of unmasked values
+    unmasked1 = ar1.compressed()
+    unmasked2 = ar2.compressed()
+    unmasked_intersect = np.intersect1d(unmasked1, unmasked2, assume_unique=assume_unique)
+
+    # Add single masked element if both inputs have masked values
+    if has_masked1 and has_masked2:
+        result_data = np.append(unmasked_intersect, ar1.fill_value)
+        result_mask = np.append(np.zeros(len(unmasked_intersect), dtype=bool), True)
+        return ma.array(result_data, mask=result_mask)
+    else:
+        return ma.array(unmasked_intersect)
-    if assume_unique:
-        aux = ma.concatenate((ar1, ar2))
-    else:
-        # Might be faster than unique( intersect1d( ar1, ar2 ) )?
-        aux = ma.concatenate((unique(ar1), unique(ar2)))
-    aux.sort()
-    return aux[:-1][aux[1:] == aux[:-1]]
```