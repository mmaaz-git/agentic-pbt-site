# Bug Report: scipy.fft.dct Type I Confusing Error Message

**Target**: `scipy.fft.dct`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

When calling `scipy.fft.dct` with `type=1` on a size-1 array, the function raises `RuntimeError: zero-length FFT requested`. This error message is confusing because the input array has length 1, not zero. The actual issue is that DCT Type I is mathematically undefined for N=1 due to division by (N-1) in its formula.

## Property-Based Test

```python
import numpy as np
import scipy.fft
from hypothesis import given, settings
from hypothesis.extra import numpy as npst


@given(npst.arrays(
    dtype=npst.floating_dtypes(),
    shape=npst.array_shapes(min_dims=1, max_dims=1, min_side=1, max_side=100)
))
@settings(max_examples=200)
def test_dct_idct_round_trip_type1(x):
    result = scipy.fft.idct(scipy.fft.dct(x, type=1), type=1)
    assert np.allclose(result, x, rtol=1e-5, atol=1e-6)
```

**Failing input**: `array([0.], dtype=float16)` (any size-1 array fails)

## Reproducing the Bug

```python
import numpy as np
import scipy.fft

x = np.array([42.0])

try:
    scipy.fft.dct(x, type=1)
except RuntimeError as e:
    print(f"Error: {e}")
```

**Output**:
```
Error: zero-length FFT requested
```

**Expected behavior**: The error message should clearly state that DCT Type I requires input length >= 2, such as:
- `"DCT Type I requires input length >= 2, got length 1"`
- `"DCT Type I is undefined for N=1 (division by N-1=0)"`

**Note**: DCT types 2, 3, and 4 all work correctly with size-1 arrays:
```python
import numpy as np
import scipy.fft

x = np.array([1.0])

for dct_type in [2, 3, 4]:
    result = scipy.fft.dct(x, type=dct_type)
    print(f"DCT type {dct_type}: {result}")
```

## Why This Is A Bug

This is a **Contract** violation because:

1. The error message is misleading - it says "zero-length FFT" when the input has length 1
2. The error doesn't explain the actual requirement that DCT Type I needs N >= 2
3. Other DCT types (2, 3, 4) work fine with N=1, making the behavior inconsistent and confusing
4. Users debugging this error waste time checking their input length (which is correct) instead of understanding the mathematical constraint

While rejecting N=1 for DCT Type I is correct (the formula has N-1 in the denominator: `π k n / (N-1)`), the error message should clearly communicate this requirement.

## Fix

The error should be raised with a clearer message. The fix would be in the scipy.fft implementation to add an explicit check for DCT Type I:

```diff
--- a/scipy/fft/_pocketfft/realtransforms.py
+++ b/scipy/fft/_pocketfft/realtransforms.py
@@ -40,6 +40,11 @@ def _r2r(x, type, s, axes, norm, out, workers, orthogonalize):
         tmp = _asfarray(x)

     if s is not None:
+        # DCT/DST Type I requires length >= 2
+        if type == 1:
+            for axis, size in zip(axes, s):
+                if size < 2:
+                    raise ValueError(f"DCT Type I requires input length >= 2 along axis {axis}, got {size}")
+
         tmp, copied = _cook_nd_args(tmp, s, axes, None)
         if copied:
             overwrite_x = True
```

Note: This is a conceptual fix. The actual implementation may require checking at a different location in the call stack. The key point is to add an explicit validation with a clear error message before the confusing "zero-length FFT" error occurs.