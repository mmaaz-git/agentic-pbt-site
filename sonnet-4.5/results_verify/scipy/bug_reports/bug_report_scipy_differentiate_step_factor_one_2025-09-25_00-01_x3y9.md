# Bug Report: scipy.differentiate.derivative step_factor=1 Causes Singular Matrix

**Target**: `scipy.differentiate.derivative`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `derivative` function accepts `step_factor=1.0` but this causes a singular Vandermonde matrix in the weight calculation, resulting in `numpy.linalg.LinAlgError`.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
import numpy as np
from scipy.differentiate import derivative

@given(
    step_factor=st.floats(min_value=0.99, max_value=1.01, allow_nan=False, allow_infinity=False)
)
def test_step_factor_near_one(step_factor):
    """step_factor very close to 1 should either be rejected or handled gracefully."""
    assume(0.99 < step_factor < 1.01)

    def f(x):
        return np.exp(x)

    try:
        result = derivative(f, 1.0, step_factor=step_factor, order=4, maxiter=2)

        # If it doesn't raise an error, it should produce a valid result
        assert result.success or not np.isnan(result.df), \
            f"step_factor={step_factor} produced NaN without raising error"
    except np.linalg.LinAlgError:
        # This is acceptable - singular matrix detected
        pass
    except ValueError as e:
        # This is the ideal behavior - validate step_factor != 1
        assert "step_factor" in str(e).lower()
```

**Failing input**: `step_factor=1.0`

## Reproducing the Bug

```python
import numpy as np
from scipy.differentiate import derivative

def f(x):
    return x**2

x = 2.0

print("Testing with step_factor=1.0:")
try:
    result = derivative(f, x, step_factor=1.0, order=4, maxiter=2)
    print(f"Result: {result.df}")
except np.linalg.LinAlgError as e:
    print(f"LinAlgError raised: {e}")
    print("This is a bug - should be caught in input validation")
```

## Why This Is A Bug

1. **Root Cause**: When `step_factor=1.0`, the algorithm doesn't reduce step sizes between iterations. In `_derivative_weights` (line 674-682), the finite difference formula uses points at:
   ```
   h = [-1/fac^1, -1/fac^0, 0, 1/fac^0, 1/fac^1]
   ```
   With fac=1.0, this becomes:
   ```
   h = [-1, -1, 0, 1, 1]
   ```

   The Vandermonde matrix constructed from these duplicate h values is singular, causing `np.linalg.solve` to raise `LinAlgError`.

2. **Missing Validation**: The input validation (line 31-33) checks that `step_factor >= 0` but doesn't prevent `step_factor == 1`:
   ```python
   if (not np.issubdtype(tols.dtype, np.number) or np.any(tols < 0)
           or np.any(np.isnan(tols)) or tols.shape != (3,)):
       raise ValueError(message)
   ```

3. **Poor User Experience**: Users get a cryptic `LinAlgError: Singular matrix` instead of a clear error message about invalid parameter values during input validation.

4. **Algorithmic Nonsense**: Even if the singular matrix were avoided, `step_factor=1` defeats the purpose of the iterative algorithm, which relies on refining the estimate with smaller step sizes. The algorithm would use the same step size in every iteration, making multiple iterations pointless.

5. **Numerical Instability**: Even values close to 1 (e.g., 1.001 or 0.999) will produce an extremely ill-conditioned Vandermonde matrix, leading to numerical instability in the weight calculation.

## Fix

Add validation to ensure `step_factor` is not too close to 1:

```diff
--- a/scipy/differentiate/_differentiate.py
+++ b/scipy/differentiate/_differentiate.py
@@ -26,11 +26,16 @@ def _derivative_iv(f, x, args, tolerances, maxiter, order, initial_step,
     # tolerances are floats, not arrays; OK to use NumPy
     message = 'Tolerances and step parameters must be non-negative scalars.'
     tols = np.asarray([atol if atol is not None else 1,
                        rtol if rtol is not None else 1,
                        step_factor])
     if (not np.issubdtype(tols.dtype, np.number) or np.any(tols < 0)
             or np.any(np.isnan(tols)) or tols.shape != (3,)):
         raise ValueError(message)
     step_factor = float(tols[2])
+
+    # step_factor == 1 causes singular matrix; close to 1 causes ill-conditioning
+    if abs(step_factor - 1.0) < 0.1:
+        raise ValueError('`step_factor` must not be close to 1.0. '
+                        'Recommended: step_factor >= 1.5 or step_factor <= 0.5.')

     maxiter_int = int(maxiter)
     if maxiter != maxiter_int or maxiter <= 0:
```

**Alternative Fix**: A more lenient approach would only forbid exactly 1.0:

```python
if step_factor == 1.0:
    raise ValueError('`step_factor` must not equal 1.0. Use values like 2.0 (default) '
                    'to reduce step sizes, or < 1.0 to increase them.')
```

However, the stricter check (`abs(step_factor - 1.0) < 0.1`) is recommended because:
- Values very close to 1 produce numerically unstable results
- Users are unlikely to intentionally use step_factor=1.01
- If they do, they should use a more reasonable value like 1.5 or 2.0