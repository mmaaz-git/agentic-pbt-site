# Bug Report: scipy.differentiate.derivative Crash with step_factor=1.0

**Target**: `scipy.differentiate.derivative`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `derivative` function crashes with `LinAlgError: Singular matrix` when `step_factor=1.0` is passed as a parameter, despite this value passing input validation.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings, assume
from scipy.differentiate import derivative
import numpy as np

@settings(max_examples=50)
@given(
    initial_step=st.floats(min_value=0.01, max_value=10, allow_nan=False, allow_infinity=False),
    step_factor=st.floats(min_value=0.1, max_value=5, allow_nan=False, allow_infinity=False)
)
def test_step_parameters_produce_valid_results(initial_step, step_factor):
    assume(step_factor > 0.05)
    x = 1.5
    res = derivative(np.exp, x, initial_step=initial_step, step_factor=step_factor)
    if res.success:
        expected = np.exp(x)
        assert math.isclose(res.df, expected, rel_tol=1e-5)
```

**Failing input**: `step_factor=1.0` (with any valid `initial_step`)

## Reproducing the Bug

```python
import numpy as np
from scipy.differentiate import derivative

res = derivative(np.exp, 1.5, step_factor=1.0)
```

Output:
```
numpy.linalg.LinAlgError: Singular matrix
```

## Why This Is A Bug

1. The input validation in `_derivative_iv` (line 31-33) only checks that `step_factor >= 0` and is not NaN, so `step_factor=1.0` passes validation.

2. The documentation does not indicate that `step_factor=1.0` is invalid. It states that `step_factor < 1` is allowed, suggesting that 1.0 should also be valid.

3. When `step_factor=1.0`, the step size remains constant across iterations. In `_derivative_weights` (line 678-682), this creates a Vandermonde matrix with repeated rows, which is singular and cannot be inverted.

4. Users receive a cryptic `LinAlgError` from NumPy's linear algebra module rather than a clear error message explaining the parameter constraint.

## Fix

```diff
diff --git a/scipy/differentiate/_differentiate.py b/scipy/differentiate/_differentiate.py
index 1234567..abcdefg 100644
--- a/scipy/differentiate/_differentiate.py
+++ b/scipy/differentiate/_differentiate.py
@@ -26,11 +26,13 @@ def _derivative_iv(f, x, args, tolerances, maxiter, order, initial_step,
     # tolerances are floats, not arrays; OK to use NumPy
     message = 'Tolerances and step parameters must be non-negative scalars.'
     tols = np.asarray([atol if atol is not None else 1,
                        rtol if rtol is not None else 1,
                        step_factor])
-    if (not np.issubdtype(tols.dtype, np.number) or np.any(tols < 0)
-            or np.any(np.isnan(tols)) or tols.shape != (3,)):
+    if (not np.issubdtype(tols.dtype, np.number) or np.any(tols < 0)
+            or np.any(np.isnan(tols)) or tols.shape != (3,) or step_factor == 1.0):
         raise ValueError(message)
+    if step_factor == 1.0:
+        raise ValueError('`step_factor` must not equal 1.0 as this creates a singular '
+                         'matrix when computing finite difference weights.')
     step_factor = float(tols[2])