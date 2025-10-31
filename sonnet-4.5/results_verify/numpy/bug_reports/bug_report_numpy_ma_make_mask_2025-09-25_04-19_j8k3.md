# Bug Report: numpy.ma.make_mask Shape Inconsistency

**Target**: `numpy.ma.make_mask`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`ma.make_mask()` inconsistently returns a scalar when the input is a single-element array containing only False, violating shape preservation guarantees.

## Property-Based Test

```python
import numpy as np
import numpy.ma as ma
from hypothesis import given, strategies as st, settings
from hypothesis.extra import numpy as npst

@st.composite
def float_masked_arrays(draw):
    shape = draw(npst.array_shapes(max_dims=1, max_side=20))
    data = draw(npst.arrays(dtype=np.float64, shape=shape,
                           elements=st.floats(allow_nan=False, allow_infinity=False,
                                            min_value=-100, max_value=100)))
    mask = draw(npst.arrays(dtype=bool, shape=shape))
    return ma.array(data, mask=mask)

@given(float_masked_arrays())
@settings(max_examples=500)
def test_make_mask_descr_consistency(arr):
    mask = ma.make_mask(ma.getmaskarray(arr))
    assert mask.shape == arr.shape
```

**Failing input**: `masked_array(data=[0.0], mask=[False])`

## Reproducing the Bug

```python
import numpy as np
import numpy.ma as ma

input_array = np.array([False])
result = ma.make_mask(input_array)

print(f"Input shape: {input_array.shape}")
print(f"Result shape: {result.shape if hasattr(result, 'shape') else 'scalar'}")
print(f"Result: {result}")

print("\nComparison with [True]:")
result_true = ma.make_mask(np.array([True]))
print(f"make_mask([True]) shape: {result_true.shape}")
print(f"make_mask([False]) shape: {result.shape if hasattr(result, 'shape') else 'scalar'}")
```

## Why This Is A Bug

`make_mask()` exhibits inconsistent behavior:
- `make_mask([True])` → `array([True])` (shape preserved)
- `make_mask([False])` → `False` (becomes scalar - shape NOT preserved)
- `make_mask([False, True])` → `array([False, True])` (shape preserved)

This violates the principle of shape preservation and creates unpredictable behavior. The same bug affects `ma.mask_or()` which returns a scalar when both inputs are `[False]`.

The root cause appears to be overly aggressive "shrinking" logic that treats `[False]` as equivalent to the scalar `nomask`/`False`, collapsing the array dimension.

## Fix

The fix should ensure that when the input is an array (even a single-element array), the output should also be an array with the same shape. The shrink parameter should control whether `nomask` is returned for all-False masks, but should not affect shape when the input is explicitly an array:

```diff
--- a/numpy/ma/core.py
+++ b/numpy/ma/core.py
@@ -1600,7 +1600,7 @@ def make_mask(m, copy=False, shrink=True, dtype=MaskType):
         result = np.array(m, dtype=MaskType, copy=copy)
     if shrink:
         result = _shrink_mask(result)
-    return result
+    return result if not isinstance(m, np.ndarray) or result.shape == m.shape else np.asarray(result, dtype=MaskType).reshape(m.shape)
```

Note: The actual fix may need to be more sophisticated to handle the shrink logic properly while preserving array shapes.