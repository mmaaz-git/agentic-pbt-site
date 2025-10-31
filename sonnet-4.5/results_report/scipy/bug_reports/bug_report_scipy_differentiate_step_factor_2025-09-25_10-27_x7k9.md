# Bug Report: scipy.differentiate.derivative Crash with step_factor Near 1.0

**Target**: `scipy.differentiate.derivative`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `derivative` function crashes with `numpy.linalg.LinAlgError: Singular matrix` when `step_factor` is at or very close to 1.0 (tested down to 1.0003), despite no documented restriction on this value.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import numpy as np
from scipy.differentiate import derivative

@given(
    step_factor=st.floats(min_value=1.0001, max_value=1.01, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=100)
def test_step_factor_close_to_one(step_factor):
    def f(x_val):
        return x_val ** 2

    x = 1.5
    res = derivative(f, x, step_factor=step_factor, maxiter=3)
```

**Failing input**: `step_factor=1.0003167180522838`

## Reproducing the Bug

```python
import numpy as np
from scipy.differentiate import derivative

def f(x_val):
    return x_val ** 2

x = 1.0

for step_factor in [1.0, 1.0003, 1.001, 1.01, 1.1]:
    try:
        res = derivative(f, x, step_factor=step_factor, maxiter=2)
        print(f"step_factor={step_factor:.4f}: SUCCESS, df={res.df:.6f}")
    except np.linalg.LinAlgError as e:
        print(f"step_factor={step_factor:.4f}: CRASH - {e}")
```

Output:
```
step_factor=1.0000: CRASH - Singular matrix
step_factor=1.0003: CRASH - Singular matrix
step_factor=1.0010: CRASH - Singular matrix
step_factor=1.0100: SUCCESS, df=2.000000
step_factor=1.1000: SUCCESS, df=2.000000
```

## Why This Is A Bug

1. **Invalid input should be rejected gracefully**: The input validation in `_derivative_iv` (lines 27-33) checks that `step_factor >= 0` but doesn't reject values at or near 1.0. Instead, the function crashes deep in the implementation with an obscure linear algebra error.

2. **Documentation doesn't mention this restriction**: The docstring describes `step_factor` as "The factor by which the step size is *reduced* in each iteration" (line 110) with no warning that values at or near 1.0 are invalid.

3. **Root cause - numerical instability**: In `_derivative_weights` (lines 674-682), the finite difference weights are computed by solving a linear system where the step sizes are `h = s / fac**p`. When `fac ≈ 1.0`:
   - All powers `fac**p` are very close to 1.0
   - The step sizes become nearly identical
   - The Vandermonde matrix `A = np.vander(h, increasing=True).T` becomes nearly singular or exactly singular
   - `np.linalg.solve(A, b)` fails

4. **User impact**: Users might set `step_factor=1.0` thinking it means "no step reduction" or due to configuration errors, and receive a cryptic linear algebra error instead of a clear validation message.

## Fix

Add validation in `_derivative_iv` to reject `step_factor` values too close to 1.0:

```diff
--- a/scipy/differentiate/_differentiate.py
+++ b/scipy/differentiate/_differentiate.py
@@ -25,11 +25,14 @@ def _derivative_iv(f, x, args, tolerances, maxiter, order, initial_step,

     # tolerances are floats, not arrays; OK to use NumPy
     message = 'Tolerances and step parameters must be non-negative scalars.'
     tols = np.asarray([atol if atol is not None else 1,
                        rtol if rtol is not None else 1,
                        step_factor])
+    if abs(step_factor - 1.0) < 0.01:
+        raise ValueError('`step_factor` must not be too close to 1.0 (must satisfy |step_factor - 1.0| >= 0.01)')
     if (not np.issubdtype(tols.dtype, np.number) or np.any(tols < 0)
             or np.any(np.isnan(tols)) or tols.shape != (3,)):
         raise ValueError(message)
     step_factor = float(tols[2])
```