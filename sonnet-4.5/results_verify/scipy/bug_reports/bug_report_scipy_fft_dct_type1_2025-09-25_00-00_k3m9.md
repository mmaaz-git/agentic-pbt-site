# Bug Report: scipy.fft.dct Type 1 Crashes on Single-Element Arrays

**Target**: `scipy.fft.dct`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

scipy.fft.dct with type=1 crashes with a confusing error message "zero-length FFT requested" when given a single-element array, while types 2, 3, and 4 handle single-element arrays correctly. The documentation does not mention any minimum length requirement for DCT type 1.

## Property-Based Test

```python
import numpy as np
import scipy.fft
from hypothesis import given, strategies as st, settings
from hypothesis.extra import numpy as st_np

@given(st_np.arrays(
    dtype=np.float64,
    shape=st_np.array_shapes(min_dims=1, max_dims=1, min_side=1, max_side=200),
    elements=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)
))
@settings(max_examples=500)
def test_dct_idct_roundtrip(x):
    for dct_type in [1, 2, 3, 4]:
        result = scipy.fft.idct(scipy.fft.dct(x, type=dct_type), type=dct_type)
        max_val = max(np.max(np.abs(x)), 1.0)
        atol = max_val * 1e-12
        assert np.allclose(result, x, rtol=1e-12, atol=atol), \
            f"DCT type {dct_type} roundtrip failed"
```

**Failing input**: `x=array([0.])`

## Reproducing the Bug

```python
import numpy as np
import scipy.fft

x = np.array([1.0])

scipy.fft.dct(x, type=1)
```

**Output:**
```
RuntimeError: zero-length FFT requested
```

**Comparison with other DCT types:**
```python
import numpy as np
import scipy.fft

x = np.array([1.0])

scipy.fft.dct(x, type=2)
scipy.fft.dct(x, type=3)
scipy.fft.dct(x, type=4)
```

All succeed with single-element arrays.

## Why This Is A Bug

This is a **contract violation** for two reasons:

1. **Undocumented restriction**: The documentation for `scipy.fft.dct` does not mention that type 1 requires a minimum array length of 2, while the other types (2, 3, 4) work fine with single-element arrays.

2. **Misleading error message**: The error "zero-length FFT requested" is confusing when the input array has length 1. A clearer error would be "DCT type 1 requires array length >= 2" or similar.

While DCT-I is mathematically defined for N >= 2 in the standard definition, the function should either:
- Document this requirement clearly
- Provide a clear error message
- Handle single elements gracefully (e.g., return the input unchanged)

## Fix

The fix should add proper input validation with a clear error message:

```diff
--- a/scipy/fft/_pocketfft/realtransforms.py
+++ b/scipy/fft/_pocketfft/realtransforms.py
@@ -42,6 +42,11 @@ def _r2r(a, func, type, s, axes, norm, overwrite_x, workers, orthogonalize):
         axes = tuple(axes)

     tmp = _asfarray(a)
+
+    # DCT/DST type 1 requires at least 2 points
+    if type == 1 and any(tmp.shape[ax] < 2 for ax in axes):
+        raise ValueError("DCT/DST type 1 requires array length of at least 2 along transformed axes")
+
     if tmp.dtype == np.float64:
         pass
     elif tmp.dtype == np.float32: