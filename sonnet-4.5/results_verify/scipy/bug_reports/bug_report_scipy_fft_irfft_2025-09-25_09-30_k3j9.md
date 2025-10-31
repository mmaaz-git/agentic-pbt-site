# Bug Report: scipy.fft.irfft Fails on Single-Element Arrays

**Target**: `scipy.fft.irfft`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`scipy.fft.irfft` raises a confusing ValueError when processing the output of `scipy.fft.rfft` for single-element arrays, breaking the round-trip property `irfft(rfft(x)) ≈ x` without an explicit `n` parameter.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy as np
import scipy.fft


@given(
    st.lists(
        st.floats(
            allow_nan=False,
            allow_infinity=False,
            min_value=-1e10,
            max_value=1e10
        ),
        min_size=1,
        max_size=1000
    )
)
def test_rfft_irfft_round_trip(data):
    x = np.array(data)
    result = scipy.fft.irfft(scipy.fft.rfft(x))
    assert np.allclose(result, x, rtol=1e-10, atol=1e-12)
```

**Failing input**: `data=[0.0]` (or any single-element array)

## Reproducing the Bug

```python
import numpy as np
import scipy.fft

x = np.array([5.0])
rfft_out = scipy.fft.rfft(x)
irfft_out = scipy.fft.irfft(rfft_out)
```

Output:
```
ValueError: Invalid number of data points (0) specified
```

## Why This Is A Bug

When `n` is not specified, `irfft` computes `n = 2*(m-1)` where `m` is the rfft output length. For single-element arrays:
- `rfft([x])` returns a length-1 array
- `irfft` calculates `n = 2*(1-1) = 0`
- This raises a confusing error: "Invalid number of data points (0) specified"

The error message is misleading because the user never specified `n=0`; the library computed it incorrectly. Additionally, while odd-length arrays require explicit `n`, single-element arrays are a valid input to `rfft`, so `irfft` should handle their output gracefully or provide a clearer error message.

## Fix

The fix should detect when the computed `n` would be invalid and provide a better error message guiding users to specify `n` explicitly:

```diff
--- a/scipy/fft/_pocketfft/basic.py
+++ b/scipy/fft/_pocketfft/basic.py
@@ -86,8 +86,12 @@ def c2r(forward, x, n=None, axis=-1, norm=None, overwrite_x=False,
     # Last axis utilizes hermitian symmetry
     if n is None:
         n = (tmp.shape[axis] - 1) * 2
-        if n < 1:
-            raise ValueError(f"Invalid number of data points ({n}) specified")
+        if n < 1:
+            raise ValueError(
+                f"Cannot infer output length from rfft/hfft input of length "
+                f"{tmp.shape[axis]}. For odd-length or single-element arrays, "
+                f"specify the output length explicitly with the 'n' parameter."
+            )
     else:
         tmp, _ = _fix_shape_1d(tmp, (n//2) + 1, axis)
```

This provides a much clearer error message that guides users to the solution.