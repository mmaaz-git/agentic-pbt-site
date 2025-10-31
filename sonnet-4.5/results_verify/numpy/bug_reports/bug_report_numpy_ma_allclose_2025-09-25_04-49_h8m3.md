# Bug Report: numpy.ma.allclose Asymmetric Behavior with Masked Values and Infinity

**Target**: `numpy.ma.allclose`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`numpy.ma.allclose(x, y)` violates the symmetry property when one array has an unmasked infinity value and the other has that position masked. Specifically, `allclose(x, y) != allclose(y, x)` in these cases, violating the fundamental mathematical property that comparison operations should be commutative.

## Property-Based Test

```python
import numpy as np
import numpy.ma as ma
from hypothesis import given, settings
import hypothesis.extra.numpy as npst

@given(npst.arrays(dtype=np.float64, shape=npst.array_shapes(max_dims=2, max_side=20)),
       npst.arrays(dtype=np.float64, shape=npst.array_shapes(max_dims=2, max_side=20)))
@settings(max_examples=500)
def test_allclose_symmetry(data1, data2):
    if data1.shape != data2.shape:
        return

    mask1 = np.random.rand(*data1.shape) < 0.3 if data1.size > 0 else np.array([])
    mask2 = np.random.rand(*data2.shape) < 0.3 if data2.size > 0 else np.array([])

    x = ma.array(data1, mask=mask1)
    y = ma.array(data2, mask=mask2)

    result_xy = ma.allclose(x, y)
    result_yx = ma.allclose(y, x)

    assert result_xy == result_yx, "allclose should be symmetric"
```

**Failing input**: Arrays where one contains unmasked infinity and the other has that position masked

## Reproducing the Bug

```python
import numpy as np
import numpy.ma as ma

x = ma.array([np.inf], mask=[False])
y = ma.array([0.], mask=[True])

print(f'allclose(x, y) = {ma.allclose(x, y)}')
print(f'allclose(y, x) = {ma.allclose(y, x)}')
```

Output:
```
allclose(x, y) = True
allclose(y, x) = False
```

## Why This Is A Bug

The `allclose` function is a comparison operation that should be symmetric: if `a` is "close to" `b`, then `b` should be "close to" `a`. This is a fundamental mathematical property that users expect from any comparison function.

The current implementation violates this when:
1. Array `x` has an unmasked infinity at position `i`
2. Array `y` has position `i` masked
3. `masked_equal=True` (the default)

In this case, `allclose(x, y)` returns `True` (because the masked position in `y` is treated as equal), but `allclose(y, x)` returns `False` (because the infinity check logic fails when the positions are swapped).

The root cause is in the infinity detection logic:

```python
xinf = np.isinf(masked_array(x, copy=False, mask=m)).filled(False)
if not np.all(xinf == filled(np.isinf(y), False)):
    return False
```

This code checks if infinities are in the same positions, but applies the combined mask `m` only to `x`, not symmetrically to both arrays.

## Fix

```diff
--- a/numpy/ma/core.py
+++ b/numpy/ma/core.py
@@ -7950,8 +7950,12 @@ def allclose(a, b, masked_equal=True, rtol=1e-5, atol=1e-8):
             y = masked_array(y, dtype=dtype, copy=False)

     m = mask_or(getmask(x), getmask(y))
-    xinf = np.isinf(masked_array(x, copy=False, mask=m)).filled(False)
-    if not np.all(xinf == filled(np.isinf(y), False)):
+    if masked_equal:
+        xinf = np.isinf(filled(x, 0)) & ~m
+        yinf = np.isinf(filled(y, 0)) & ~m
+    else:
+        xinf = np.isinf(filled(x, 0))
+        yinf = np.isinf(filled(y, 0))
+    if not np.all(xinf == yinf):
         return False
```

This fix ensures that when `masked_equal=True`, masked positions are excluded from the infinity comparison symmetrically for both arrays, restoring the symmetry property.