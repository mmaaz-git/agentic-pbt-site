# Bug Report: scipy.signal.windows.kaiser NaN with Large Beta Values

**Target**: `scipy.signal.windows.kaiser`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `kaiser` window function returns NaN values when beta is large (>= 710) and M is small, due to numerical overflow in the Bessel function calculation. This violates the expectation that window functions should return finite values for all valid inputs.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import numpy as np
import scipy.signal.windows as windows

@given(
    M=st.integers(min_value=2, max_value=20),
    beta=st.floats(min_value=100.0, max_value=1000.0, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=30)
def test_kaiser_very_large_beta(M, beta):
    window = windows.kaiser(M, beta, sym=True)

    assert len(window) == M
    assert np.all(np.isfinite(window)), \
        f"Kaiser window should have finite values even for large beta"
```

**Failing input**: `M=3, beta=710.0`

## Reproducing the Bug

```python
import numpy as np
import scipy.signal.windows as windows

M = 3
beta = 710.0
window = windows.kaiser(M, beta)

print(f"windows.kaiser({M}, {beta}) = {window}")

assert np.all(np.isfinite(window)), "Kaiser window should not contain NaN values"
```

**Output**:
```
windows.kaiser(3, 710.0) = [ 0. nan  0.]
RuntimeWarning: invalid value encountered in divide
  w = (special.i0(beta * xp.sqrt(1 - ((n - alpha) / alpha) ** 2.0)) /
AssertionError: Kaiser window should not contain NaN values
```

## Why This Is A Bug

1. **Silent failure**: The function returns NaN values without raising an exception, making it easy for users to unknowingly propagate NaN values through their calculations.

2. **Violates documented behavior**: The documentation states the function returns "The window, with the maximum value normalized to 1", not "The window, or NaN values if parameters are too extreme".

3. **Inconsistent with scipy conventions**: Most scipy functions either handle edge cases gracefully or raise appropriate exceptions for invalid inputs.

4. **While documented, it's inadequate**: The docstring mentions "as beta gets large, the window narrows, and so the number of samples needs to be large enough to sample the increasingly narrow spike, otherwise NaNs will be returned" (line 1244), but this:
   - Doesn't specify what values cause NaNs
   - Doesn't provide validation or warnings
   - Is buried in the notes rather than being checked at runtime

## Impact

- Users may unknowingly pass large beta values and get NaN results
- Scientific computations using kaiser windows may silently fail
- Automated processing pipelines may break unexpectedly

## Fix

The function should either:

1. **Validate inputs** and raise `ValueError` with a clear message when beta is too large for the given M
2. **Clamp the output** to prevent NaN values (e.g., using `np.nan_to_num`)
3. **Use more numerically stable computation** for large beta values

**Suggested validation**:

```diff
def kaiser(M, beta, sym=True, *, xp=None, device=None):
    xp = _namespace(xp)

    if _len_guards(M):
        return xp.ones(M, dtype=xp.float64, device=device)
+
+   # Prevent numerical overflow that causes NaN values
+   # For small M and large beta, the Bessel function overflows
+   if beta > 700 and M < 10:
+       raise ValueError(
+           f"beta={beta} is too large for M={M}. "
+           f"For M < 10, beta should be < 700 to avoid numerical overflow. "
+           f"Either increase M or decrease beta."
+       )

    M, needs_trunc = _extend(M, sym)
    n = xp.arange(0, M, dtype=xp.float64, device=device)
    alpha = (M - 1) / 2.0
    w = (special.i0(beta * xp.sqrt(1 - ((n - alpha) / alpha) ** 2.0)) /
         special.i0(xp.asarray(beta, dtype=xp.float64)))

    return _truncate(w, needs_trunc)
```

Alternatively, use log-domain Bessel functions to avoid overflow:

```diff
def kaiser(M, beta, sym=True, *, xp=None, device=None):
    xp = _namespace(xp)

    if _len_guards(M):
        return xp.ones(M, dtype=xp.float64, device=device)

    M, needs_trunc = _extend(M, sym)
    n = xp.arange(0, M, dtype=xp.float64, device=device)
    alpha = (M - 1) / 2.0
-   w = (special.i0(beta * xp.sqrt(1 - ((n - alpha) / alpha) ** 2.0)) /
-        special.i0(xp.asarray(beta, dtype=xp.float64)))
+
+   # Use log-domain to prevent overflow for large beta
+   arg = beta * xp.sqrt(1 - ((n - alpha) / alpha) ** 2.0)
+   log_i0_arg = special.log_ndtr(arg)  # or use a custom log_i0 function
+   log_i0_beta = special.log_ndtr(beta)
+   w = xp.exp(log_i0_arg - log_i0_beta)

    return _truncate(w, needs_trunc)
```