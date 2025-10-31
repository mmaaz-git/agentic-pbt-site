# Bug Report: numpy.ma.allclose Violates Symmetry Property with Masked Infinity Values

**Target**: `numpy.ma.allclose`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `numpy.ma.allclose` function violates the mathematical property of symmetry when comparing arrays where one contains an unmasked infinity value and the other has that position masked, returning different results depending on argument order.

## Property-Based Test

```python
import numpy as np
import numpy.ma as ma
from hypothesis import given, settings, example
import hypothesis.extra.numpy as npst

@given(npst.arrays(dtype=np.float64, shape=npst.array_shapes(max_dims=2, max_side=20)),
       npst.arrays(dtype=np.float64, shape=npst.array_shapes(max_dims=2, max_side=20)))
@settings(max_examples=500)
@example(np.array([np.inf]), np.array([0.]))
def test_allclose_symmetry(data1, data2):
    if data1.shape != data2.shape:
        return

    mask1 = np.random.rand(*data1.shape) < 0.3 if data1.size > 0 else np.array([])
    mask2 = np.random.rand(*data2.shape) < 0.3 if data2.size > 0 else np.array([])

    x = ma.array(data1, mask=mask1)
    y = ma.array(data2, mask=mask2)

    result_xy = ma.allclose(x, y)
    result_yx = ma.allclose(y, x)

    assert result_xy == result_yx, f"allclose should be symmetric but ma.allclose(x, y)={result_xy} != ma.allclose(y, x)={result_yx}\nx={x}\ny={y}"

if __name__ == "__main__":
    test_allclose_symmetry()
```

<details>

<summary>
**Failing input**: `data1=array([0., 0., ..., 0., 0.]), data2=array([0., 0., ..., 0., inf])`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/19/hypo.py", line 26, in <module>
    test_allclose_symmetry()
    ~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/19/hypo.py", line 7, in test_allclose_symmetry
    npst.arrays(dtype=np.float64, shape=npst.array_shapes(max_dims=2, max_side=20)))
           ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/19/hypo.py", line 23, in test_allclose_symmetry
    assert result_xy == result_yx, f"allclose should be symmetric but ma.allclose(x, y)={result_xy} != ma.allclose(y, x)={result_yx}\nx={x}\ny={y}"
           ^^^^^^^^^^^^^^^^^^^^^^
AssertionError: allclose should be symmetric but ma.allclose(x, y)=False != ma.allclose(y, x)=True
x=[0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 --]
y=[-- -- 0.0 0.0 0.0 0.0 0.0 0.0 0.0 -- 0.0 -- 0.0 0.0 inf]
Falsifying example: test_allclose_symmetry(
    data1=array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
    data2=array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
            0., inf]),
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
import numpy.ma as ma

# Create test arrays
x = ma.array([np.inf], mask=[False])
y = ma.array([0.], mask=[True])

print("Testing numpy.ma.allclose asymmetry with masked values and infinity:")
print(f"x = ma.array([np.inf], mask=[False])")
print(f"y = ma.array([0.], mask=[True])")
print()
print(f"ma.allclose(x, y) = {ma.allclose(x, y)}")
print(f"ma.allclose(y, x) = {ma.allclose(y, x)}")
print()
print("Expected: Both should return the same value (symmetric)")
print("Actual: They return different values (asymmetric)")
```

<details>

<summary>
AssertionError: allclose(x, y) != allclose(y, x)
</summary>
```
Testing numpy.ma.allclose asymmetry with masked values and infinity:
x = ma.array([np.inf], mask=[False])
y = ma.array([0.], mask=[True])

ma.allclose(x, y) = True
ma.allclose(y, x) = False

Expected: Both should return the same value (symmetric)
Actual: They return different values (asymmetric)
```
</details>

## Why This Is A Bug

The `numpy.ma.allclose` function violates the fundamental mathematical property that comparison operations should be symmetric. While the parent function `numpy.allclose` is documented to be potentially asymmetric due to the tolerance formula `|a - b| <= (atol + rtol * |b|)`, the bug here represents a different type of asymmetry that occurs specifically when comparing masked arrays with infinity values.

The issue arises in the infinity checking logic (lines 8552-8554 of numpy/ma/core.py):
- When `masked_equal=True` (default), masked values should be treated as "equal" to anything
- However, the infinity check applies the combined mask asymmetrically to only the first array
- This causes `allclose([inf], [masked])` to return `True` (masked treated as matching)
- But `allclose([masked], [inf])` returns `False` (infinity check fails)

This behavior contradicts the documented purpose of `masked_equal=True`, which states that masked values are "treated as equal." If masked values truly match anything when `masked_equal=True`, the comparison should be symmetric regardless of which array contains the masked value.

## Relevant Context

The bug is located in `/home/npc/miniconda/lib/python3.13/site-packages/numpy/ma/core.py` around lines 8551-8554. The problematic code:

```python
m = mask_or(getmask(x), getmask(y))
xinf = np.isinf(masked_array(x, copy=False, mask=m)).filled(False)
# If we have some infs, they should fall at the same place.
if not np.all(xinf == filled(np.isinf(y), False)):
    return False
```

The issue is that the combined mask `m` is applied to `x` when checking for infinities, but not symmetrically applied to `y`. This breaks symmetry when one array has unmasked infinity and the other has that position masked.

Documentation reference: https://numpy.org/doc/stable/reference/generated/numpy.ma.allclose.html

## Proposed Fix

```diff
--- a/numpy/ma/core.py
+++ b/numpy/ma/core.py
@@ -8550,9 +8550,13 @@ def allclose(a, b, masked_equal=True, rtol=1e-5, atol=1e-8):
             y = masked_array(y, dtype=dtype, copy=False)

     m = mask_or(getmask(x), getmask(y))
-    xinf = np.isinf(masked_array(x, copy=False, mask=m)).filled(False)
-    # If we have some infs, they should fall at the same place.
-    if not np.all(xinf == filled(np.isinf(y), False)):
+    if masked_equal:
+        # When masked_equal=True, exclude masked positions from infinity check
+        xinf = np.isinf(filled(x, 0)) & ~m
+        yinf = np.isinf(filled(y, 0)) & ~m
+    else:
+        xinf = np.isinf(filled(x, 0))
+        yinf = np.isinf(filled(y, 0))
+    if not np.all(xinf == yinf):
         return False
     # No infs at all
     if not np.any(xinf):
```