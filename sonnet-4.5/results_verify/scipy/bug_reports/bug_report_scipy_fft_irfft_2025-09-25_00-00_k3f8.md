# Bug Report: scipy.fft.irfft Crashes on Single-Element Input

**Target**: `scipy.fft.irfft`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`scipy.fft.irfft` crashes with a ValueError when given the output of `rfft` applied to a single-element array, despite single-element arrays being valid input to `rfft`.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from hypothesis.extra.numpy import arrays
import numpy as np
import scipy.fft

float_strategy = st.floats(
    min_value=-1e8,
    max_value=1e8,
    allow_nan=False,
    allow_infinity=False
)

@given(arrays(dtype=np.float64, shape=st.integers(min_value=1, max_value=100), elements=float_strategy))
def test_rfft_irfft_round_trip(x):
    """Round-trip property for real FFT: irfft(rfft(x)) should equal x"""
    result = scipy.fft.irfft(scipy.fft.rfft(x))
    scale = max(np.max(np.abs(x)), 1.0)
    np.testing.assert_allclose(result, x, rtol=1e-10, atol=scale * 1e-10)
```

**Failing input**: `x=array([0.])`

## Reproducing the Bug

```python
import numpy as np
import scipy.fft

x = np.array([5.])
rfft_result = scipy.fft.rfft(x)
result = scipy.fft.irfft(rfft_result)
```

**Output**:
```
ValueError: Invalid number of data points (0) specified
```

## Why This Is A Bug

The documentation for `irfft` states that when `n` is not given, it defaults to `2*(m-1)` where `m` is the length of the input. For a single-element input from `rfft`, this evaluates to `2*(1-1) = 0`, which is invalid.

This violates the expected behavior that `irfft(rfft(x), len(x))` should work for all valid inputs. While the workaround is to explicitly pass `n=1`, the function should either:

1. Handle the single-element case gracefully
2. Provide a more informative error message
3. Document that single-element inputs are not supported
4. Use a better default that avoids `n=0`

The crash is particularly problematic because:
- `rfft` accepts single-element arrays without issue
- The error message "Invalid number of data points (0) specified" is cryptic and doesn't explain the root cause
- Users would reasonably expect `irfft(rfft(x))` to work for any valid input `x`

## Fix

The issue is in the default value calculation for `n`. The code should check if the calculated default would be 0 or negative, and either raise a more informative error or use a minimum value of 1.

```diff
--- a/scipy/fft/_pocketfft/basic.py
+++ b/scipy/fft/_pocketfft/basic.py
@@ -85,6 +85,9 @@ def c2r(forward, x, n, axis, norm, overwrite_x, workers, plan):
     if n is None:
         n = 2 * (x.shape[axis] - 1)

+    if n <= 0:
+        raise ValueError(f"Cannot compute irfft: input length {x.shape[axis]} is too small. "
+                        f"For single-element input, explicitly specify n parameter.")
+
     if n < 1:
         raise ValueError(f"Invalid number of data points ({n}) specified")
```

This provides a clearer error message that guides users to the solution.