# Bug Report: scipy.differentiate.derivative step_factor Near 1.0 Crash

**Target**: `scipy.differentiate.derivative`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `derivative` function crashes with `numpy.linalg.LinAlgError: Singular matrix` when `step_factor` is 1.0 or very close to 1.0 (e.g., 1.0003), even though the documentation has no restriction against such values.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import numpy as np
from scipy.differentiate import derivative

@given(x=st.floats(min_value=0.5, max_value=2, allow_nan=False, allow_infinity=False))
@settings(max_examples=100)
def test_step_factor_exactly_one(x):
    def f(x_val):
        return x_val ** 2

    res = derivative(f, x, step_factor=1.0, maxiter=5)

    if res.success:
        assert abs(res.df - 2 * x) < 1e-6
```

**Failing input**: `x=1.0` (or any value)

## Reproducing the Bug

```python
import numpy as np
from scipy.differentiate import derivative

def f(x_val):
    return x_val ** 2

x = 1.0

try:
    res = derivative(f, x, step_factor=1.0, maxiter=2)
    print(f"Result: {res.df}")
except np.linalg.LinAlgError as e:
    print(f"Crash: {e}")
```

Output:
```
Crash: Singular matrix
```

## Why This Is A Bug

1. **Undocumented restriction**: The documentation for `step_factor` states it is "The factor by which the step size is *reduced* in each iteration" with no mention that `step_factor=1.0` is invalid. The input validation in `_derivative_iv` only checks that `step_factor >= 0`, allowing 1.0.

2. **Invalid user input should be handled gracefully**: Even if `step_factor=1.0` is not mathematically useful (it means the step size never changes), the function should either:
   - Validate and raise a clear `ValueError` with an informative message
   - Handle the edge case without crashing

3. **Root cause**: When `step_factor=1.0`, all step sizes `h = s / fac**p` become identical (since `1.0**p = 1.0` for any p). This creates a Vandermonde matrix with duplicate rows, making it singular. The `np.linalg.solve` call at line 682 then fails.

4. **User impact**: Users experimenting with parameters or using configuration files might accidentally set `step_factor=1.0` and encounter an obscure linear algebra error rather than a clear validation error.

## Fix

Add input validation to reject `step_factor=1.0`:

```diff
--- a/scipy/differentiate/_differentiate.py
+++ b/scipy/differentiate/_differentiate.py
@@ -30,7 +30,7 @@ def _derivative_iv(f, x, args, tolerances, maxiter, order, initial_step,
     tols = np.asarray([atol if atol is not None else 1,
                        rtol if rtol is not None else 1,
                        step_factor])
-    if (not np.issubdtype(tols.dtype, np.number) or np.any(tols < 0)
+    if (not np.issubdtype(tols.dtype, np.number) or np.any(tols < 0) or step_factor == 1.0
             or np.any(np.isnan(tols)) or tols.shape != (3,)):
         raise ValueError(message)
     step_factor = float(tols[2])
```

Alternatively, update the error message to be more specific:

```diff
--- a/scipy/differentiate/_differentiate.py
+++ b/scipy/differentiate/_differentiate.py
@@ -26,7 +26,10 @@ def _derivative_iv(f, x, args, tolerances, maxiter, order, initial_step,
     # tolerances are floats, not arrays; OK to use NumPy
     message = 'Tolerances and step parameters must be non-negative scalars.'
     tols = np.asarray([atol if atol is not None else 1,
                        rtol if rtol is not None else 1,
                        step_factor])
+    if step_factor == 1.0:
+        raise ValueError('`step_factor` must not equal 1.0.')
     if (not np.issubdtype(tols.dtype, np.number) or np.any(tols < 0)
             or np.any(np.isnan(tols)) or tols.shape != (3,)):
         raise ValueError(message)
```