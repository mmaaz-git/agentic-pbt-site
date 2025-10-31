# Bug Report: scipy.differentiate.derivative Invalid initial_step Handling

**Target**: `scipy.differentiate.derivative`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `derivative` function silently accepts non-positive `initial_step` values (≤ 0) and produces NaN results instead of raising a descriptive error during input validation, violating API contract expectations.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy as np
from scipy.differentiate import derivative

@given(
    initial_step=st.floats(min_value=-10.0, max_value=0.0, allow_nan=False, allow_infinity=False)
)
def test_initial_step_validation(initial_step):
    """initial_step must be positive - function should raise ValueError for invalid values."""
    def f(x):
        return x**2

    try:
        result = derivative(f, 1.0, initial_step=initial_step)
        # If we get here without exception, check if result is valid
        assert not np.isnan(result.df), \
            f"Function returned NaN for initial_step={initial_step} instead of raising error"
        assert False, \
            f"Function accepted invalid initial_step={initial_step} without raising ValueError"
    except ValueError as e:
        # This is the expected behavior
        assert "initial_step" in str(e).lower() or "step" in str(e).lower(), \
            f"ValueError raised but message doesn't mention step parameter: {e}"
```

**Failing input**: `initial_step=0.0` or any negative value

## Reproducing the Bug

```python
import numpy as np
from scipy.differentiate import derivative

def f(x):
    return x**2

x = 2.0
true_derivative = 4.0

print("Test 1: initial_step = 0")
result = derivative(f, x, initial_step=0.0)
print(f"  Result: df={result.df}, status={result.status}, success={result.success}")
print(f"  Expected: ValueError with message about invalid initial_step")
print(f"  Actual: Silently returns NaN={np.isnan(result.df)}")

print("\nTest 2: initial_step = -0.5")
result = derivative(f, x, initial_step=-0.5)
print(f"  Result: df={result.df}, status={result.status}, success={result.success}")
print(f"  Expected: ValueError with message about invalid initial_step")
print(f"  Actual: Silently returns NaN={np.isnan(result.df)}")

print("\nTest 3: initial_step = 0.5 (valid, control)")
result = derivative(f, x, initial_step=0.5)
print(f"  Result: df={result.df}, error={abs(result.df - true_derivative)}")
print(f"  Works correctly")
```

## Why This Is A Bug

This violates the API contract and user expectations in several ways:

1. **Missing Input Validation**: The function validates that `step_factor` is non-negative (line 31-33 in `_differentiate.py`), but fails to validate `initial_step` similarly. This is inconsistent.

2. **Silent Failure**: Instead of raising a descriptive `ValueError` during input validation, the code silently converts non-positive step sizes to NaN at line 408:
   ```python
   h0 = xpx.at(h0)[h0 <= 0].set(xp.nan)
   ```
   This NaN then propagates through the computation, producing a NaN result with `success=False` and `status=-3` (non-finite value encountered).

3. **Poor Error Message**: When the user gets `status=-3`, the documentation says "A non-finite value was encountered" - this suggests the *function* returned NaN, not that the user provided an invalid parameter. This makes debugging difficult.

4. **Inconsistent with SciPy Conventions**: Most SciPy functions validate input parameters and raise descriptive errors (e.g., `ValueError: initial_step must be positive`) rather than silently producing NaN results.

5. **Documentation Mismatch**: The docstring states `initial_step` is "The (absolute) initial step size" with default 0.5, suggesting it should be positive, but doesn't explicitly state this constraint or what happens if violated.

## Fix

Add explicit validation for `initial_step` in the `_derivative_iv` function, similar to how `step_factor` is validated:

```diff
--- a/scipy/differentiate/_differentiate.py
+++ b/scipy/differentiate/_differentiate.py
@@ -28,8 +28,9 @@ def _derivative_iv(f, x, args, tolerances, maxiter, order, initial_step,
     tols = np.asarray([atol if atol is not None else 1,
                        rtol if rtol is not None else 1,
-                       step_factor])
-    if (not np.issubdtype(tols.dtype, np.number) or np.any(tols < 0)
+                       step_factor,
+                       initial_step])
+    if (not np.issubdtype(tols.dtype, np.number) or np.any(tols < 0)
             or np.any(np.isnan(tols)) or tols.shape != (3,)):
         raise ValueError(message)
     step_factor = float(tols[2])
@@ -44,7 +45,7 @@ def _derivative_iv(f, x, args, tolerances, maxiter, order, initial_step,
     step_direction = xp.asarray(step_direction)
-    initial_step = xp.asarray(initial_step)
+    initial_step = xp.asarray(initial_step, dtype=xp.float64)
     temp = xp.broadcast_arrays(x, step_direction, initial_step)
     x, step_direction, initial_step = temp
```

Wait, that's not quite right. The issue is that `initial_step` can be an array, so we need to validate it after broadcasting. A better fix:

```diff
--- a/scipy/differentiate/_differentiate.py
+++ b/scipy/differentiate/_differentiate.py
@@ -44,6 +44,11 @@ def _derivative_iv(f, x, args, tolerances, maxiter, order, initial_step,

     step_direction = xp.asarray(step_direction)
     initial_step = xp.asarray(initial_step)
     temp = xp.broadcast_arrays(x, step_direction, initial_step)
     x, step_direction, initial_step = temp
+
+    # Validate initial_step is positive
+    if np.any(np.asarray(initial_step) <= 0):
+        raise ValueError('`initial_step` must contain only positive values.')

     message = '`preserve_shape` must be True or False.'
```

This fix:
1. Validates that all elements of `initial_step` are positive after broadcasting
2. Raises a clear `ValueError` with a descriptive message
3. Occurs during input validation (before any computation)
4. Is consistent with how other parameters are validated