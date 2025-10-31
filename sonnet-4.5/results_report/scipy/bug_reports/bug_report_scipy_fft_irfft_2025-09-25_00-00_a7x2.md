# Bug Report: scipy.fft.irfft Single-Element Input Crash

**Target**: `scipy.fft.irfft`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`scipy.fft.irfft()` crashes with a ValueError when given a single-element complex array as input, even though such arrays are valid outputs from `scipy.fft.rfft()`.

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
def test_rfft_irfft_roundtrip(data):
    x = np.array(data)
    result = scipy.fft.irfft(scipy.fft.rfft(x))
    np.testing.assert_allclose(result, x, rtol=1e-10, atol=1e-10)
```

**Failing input**: `[0.0]` (and any other single-element array)

## Reproducing the Bug

```python
import numpy as np
import scipy.fft

x = np.array([5.0])
fft_result = scipy.fft.rfft(x)
print(f"rfft result: {fft_result}")

try:
    roundtrip = scipy.fft.irfft(fft_result)
except ValueError as e:
    print(f"Error: {e}")
```

Output:
```
rfft result: [5.+0.j]
Error: Invalid number of data points (0) specified
```

## Why This Is A Bug

1. Single-element arrays are valid inputs to `scipy.fft.rfft()` and produce single-element complex array outputs
2. The inverse function `irfft()` should accept the output of `rfft()` as valid input
3. The function crashes instead of handling this edge case gracefully
4. The workaround `scipy.fft.irfft(fft_result, n=1)` works, proving the operation is mathematically valid

The issue is in `/scipy/fft/_pocketfft/basic.py:87-90`:

```python
if n is None:
    n = (tmp.shape[axis] - 1) * 2
    if n < 1:
        raise ValueError(f"Invalid number of data points ({n}) specified")
```

For single-element input, `n = (1-1)*2 = 0`, which triggers the ValueError.

## Fix

```diff
--- a/scipy/fft/_pocketfft/basic.py
+++ b/scipy/fft/_pocketfft/basic.py
@@ -85,8 +85,8 @@ def c2r(forward, x, n=None, axis=-1, norm=None, overwrite_x=False,

     # Last axis utilizes hermitian symmetry
     if n is None:
-        n = (tmp.shape[axis] - 1) * 2
-        if n < 1:
+        n = max(1, (tmp.shape[axis] - 1) * 2)
+        if tmp.shape[axis] < 1:
             raise ValueError(f"Invalid number of data points ({n}) specified")
     else:
         tmp, _ = _fix_shape_1d(tmp, (n//2) + 1, axis)
```

This fix ensures that `n` is at least 1 for single-element inputs, which is the correct behavior for the inverse transform of a single-element FFT result.