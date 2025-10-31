# Bug Report: scipy.differentiate.derivative Missing initial_step Validation

**Target**: `scipy.differentiate.derivative`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `derivative` function fails to validate that `initial_step` must be positive, leading to silent failures when non-positive values are provided. Instead of raising a clear `ValueError` during input validation, the function sets invalid step sizes to NaN and proceeds with computation, resulting in confusing failure modes.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from scipy.differentiate import derivative
import numpy as np

@given(st.floats(min_value=-10, max_value=0, allow_nan=False, allow_infinity=False))
def test_derivative_should_reject_non_positive_initial_step(initial_step):
    with pytest.raises(ValueError, match="initial_step"):
        derivative(np.sin, 1.0, initial_step=initial_step)
```

**Failing input**: `initial_step=0.0` (or any non-positive value)

## Reproducing the Bug

```python
import numpy as np
from scipy.differentiate import derivative

result = derivative(np.sin, 1.0, initial_step=0.0)
print(f"success: {result.success}")
print(f"status: {result.status}")
print(f"df: {result.df}")
```

Expected: `ValueError` with message about invalid `initial_step`
Actual: No error raised, function proceeds with NaN step sizes

## Why This Is A Bug

The function documentation does not specify that `initial_step` must be positive, but:

1. The error message at line 27 states "Tolerances and step parameters must be non-negative scalars"
2. Line 408 explicitly handles non-positive step sizes by setting them to NaN: `h0 = xpx.at(h0)[h0 <= 0].set(xp.nan)`
3. This indicates the function expects positive step sizes but fails to validate this in the input validation function `_derivative_iv`

The validation for `step_factor` exists (lines 27-34) but `initial_step` is never validated, violating the principle of fail-fast and clear error messages.

## Fix

```diff
diff --git a/scipy/differentiate/_differentiate.py b/scipy/differentiate/_differentiate.py
index 1234567..abcdefg 100644
--- a/scipy/differentiate/_differentiate.py
+++ b/scipy/differentiate/_differentiate.py
@@ -44,6 +44,13 @@ def _derivative_iv(f, x, args, tolerances, maxiter, order, initial_step,
     step_direction = xp.asarray(step_direction)
     initial_step = xp.asarray(initial_step)
     temp = xp.broadcast_arrays(x, step_direction, initial_step)
     x, step_direction, initial_step = temp
+
+    # Validate that initial_step is positive
+    message = '`initial_step` must contain only positive values.'
+    if xp.any(initial_step <= 0) or xp.any(xp.isnan(initial_step)):
+        raise ValueError(message)

     message = '`preserve_shape` must be True or False.'
     if preserve_shape not in {True, False}:
```