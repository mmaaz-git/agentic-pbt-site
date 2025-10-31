# Bug Report: numpy.array_equiv Violates Reflexivity with NaN

**Target**: `numpy.array_equiv`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`numpy.array_equiv(arr, arr)` returns `False` when `arr` contains NaN values, violating the reflexivity property that any array should be equivalent to itself.

## Property-Based Test

```python
from hypothesis import given
import hypothesis.extra.numpy as npst
import numpy as np

@given(npst.arrays(dtype=npst.floating_dtypes(), shape=npst.array_shapes()))
def test_array_equiv_reflexivity(arr):
    assert np.array_equiv(arr, arr)
```

**Failing input**: `array([nan], dtype=float16)`

## Reproducing the Bug

```python
import numpy as np

arr = np.array([np.nan])
result = np.array_equiv(arr, arr)
print(f"np.array_equiv(arr, arr) = {result}")
```

Output:
```
np.array_equiv(arr, arr) = False
```

Expected: `True` (reflexivity - an array should always be equivalent to itself)

## Why This Is A Bug

The function `array_equiv` is supposed to test if two arrays are equivalent. Mathematically, equivalence relations must satisfy reflexivity: any element must be equivalent to itself. However, `np.array_equiv(arr, arr)` returns `False` when `arr` contains NaN values.

This violates user expectations because:
1. An array should always be equivalent to itself, regardless of its contents
2. The related function `array_equal` provides an `equal_nan` parameter to handle this case correctly
3. This violates the mathematical definition of an equivalence relation

## Fix

Add an `equal_nan` parameter to `array_equiv` similar to `array_equal`, or handle the reflexivity case explicitly:

```diff
--- a/numpy/_core/numeric.py
+++ b/numpy/_core/numeric.py
@@ -2614,7 +2614,7 @@ def _array_equiv_dispatcher(a1, a2):

 @array_function_dispatch(_array_equiv_dispatcher)
-def array_equiv(a1, a2):
+def array_equiv(a1, a2, equal_nan=True):
     """
     Returns True if input arrays are shape consistent and all elements equal.

@@ -2624,6 +2624,9 @@ def array_equiv(a1, a2):
     a1, a2 : array_like
         Input arrays.
+    equal_nan : bool, optional
+        Whether to compare NaN's as equal. If True, NaN's in `a1` will be
+        considered equal to NaN's in `a2`. Default is True.

     Returns
     -------
@@ -2657,7 +2660,17 @@ def array_equiv(a1, a2):
     except Exception:
         return False

-    return builtins.bool(asanyarray(a1 == a2).all())
+    if not equal_nan:
+        return builtins.bool(asanyarray(a1 == a2).all())
+
+    # Handle NaN values when equal_nan is True
+    eq = (a1 == a2)
+    if not eq.all():
+        # Check if all differences are due to NaN
+        a1_isnan = isnan(a1)
+        a2_isnan = isnan(a2)
+        eq = eq | (a1_isnan & a2_isnan)
+
+    return builtins.bool(asanyarray(eq).all())

```

Note: Setting `equal_nan=True` as the default maintains reflexivity while potentially changing existing behavior. An alternative is to set it to `False` by default but explicitly handle the reflexivity case when `a1 is a2`.