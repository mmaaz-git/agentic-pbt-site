# Bug Report: xarray.plot.utils._rescale_imshow_rgb Missing Validation

**Target**: `xarray.plot.utils._rescale_imshow_rgb`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_rescale_imshow_rgb` function fails to validate that `vmin < vmax` when both parameters are explicitly provided, leading to silent incorrect behavior with inverted color scaling.

## Property-Based Test

```python
from hypothesis import given, assume, strategies as st
import numpy as np
from xarray.plot.utils import _rescale_imshow_rgb
import pytest

@given(
    st.floats(allow_nan=False, allow_infinity=False, min_value=0, max_value=1e6),
    st.floats(allow_nan=False, allow_infinity=False, min_value=0, max_value=1e6)
)
def test_rescale_imshow_rgb_vmin_greater_than_vmax_rejected(vmin, vmax):
    assume(vmin > vmax)
    darray = np.random.uniform(0, 100, (10, 10, 3)).astype('f8')

    with pytest.raises(ValueError):
        _rescale_imshow_rgb(darray, vmin=vmin, vmax=vmax, robust=False)
```

**Failing input**: `vmin=1.0, vmax=0.0`

## Reproducing the Bug

```python
import numpy as np
from xarray.plot.utils import _rescale_imshow_rgb

darray = np.array([[[50.0, 50.0, 50.0]]]).astype('f8')

result = _rescale_imshow_rgb(darray, vmin=100.0, vmax=0.0, robust=False)
print(result)
```

When `vmin > vmax`, the function should raise a `ValueError` but instead proceeds with the calculation, producing incorrect results through the formula `(darray - vmin) / (vmax - vmin)` where the denominator is negative.

## Why This Is A Bug

The function validates `vmin < vmax` when one parameter is None (lines 762-766 and 769-773), but completely skips this validation when both parameters are explicitly provided. This inconsistency leads to silent failures where users specify inverted bounds and get incorrectly scaled images without any warning.

The code at lines 760-773 has three branches:
1. If `vmax is None`: validate that the default vmax > provided vmin
2. If `vmin is None`: validate that provided vmax > default vmin (0)
3. If both are provided: **no validation** (bug!)

## Fix

```diff
--- a/xarray/plot/utils.py
+++ b/xarray/plot/utils.py
@@ -771,6 +771,11 @@ def _rescale_imshow_rgb(darray, vmin, vmax, robust):
                 f"vmax={vmax!r} is less than the default vmin (0) - you must supply "
                 "a vmin < vmax in this case."
             )
+    else:
+        # Both vmin and vmax are provided, validate they form a valid interval
+        if vmin >= vmax:
+            raise ValueError(
+                f"vmin ({vmin!r}) must be less than vmax ({vmax!r})."
+            )
     # Scale interval [vmin .. vmax] to [0 .. 1], with darray as 64-bit float
     # to avoid precision loss, integer over/underflow, etc with extreme inputs.
     # After scaling, downcast to 32-bit float.  This substantially reduces