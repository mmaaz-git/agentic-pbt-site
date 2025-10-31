# Bug Report: scipy.differentiate.derivative initial_step validation

**Target**: `scipy.differentiate.derivative`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `derivative` function accepts `initial_step=0` and negative `initial_step` values without raising an error during input validation, but these values cause the function to fail silently, returning `success=False` and NaN results. The docstring indicates initial_step should be "(absolute)" suggesting it must be positive, but this is not validated.

## Property-Based Test

```python
import numpy as np
import scipy.differentiate
from hypothesis import given, strategies as st, settings
import pytest


@given(
    x=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=100)
def test_derivative_zero_initial_step_error(x):
    """
    Property: initial_step=0 should raise an error during input validation.
    """
    def f(x_val):
        return x_val**2

    with pytest.raises(ValueError):
        scipy.differentiate.derivative(f, x, initial_step=0)


@given(
    x=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
    initial_step=st.floats(min_value=-10, max_value=-0.001, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=100)
def test_derivative_negative_initial_step_error(x, initial_step):
    """
    Property: negative initial_step should raise an error.
    """
    def f(x_val):
        return x_val**2

    with pytest.raises(ValueError):
        scipy.differentiate.derivative(f, x, initial_step=initial_step)
```

**Failing input**: `x=0.0, initial_step=0` or `x=0.0, initial_step=-0.5`

## Reproducing the Bug

```python
import numpy as np
import scipy.differentiate

def f(x):
    return x**2

x = 1.0

result = scipy.differentiate.derivative(f, x, initial_step=0)
print(f"initial_step=0:")
print(f"  success: {result.success}")
print(f"  df: {result.df}")

result2 = scipy.differentiate.derivative(f, x, initial_step=-0.5)
print(f"\ninitial_step=-0.5:")
print(f"  success: {result2.success}")
print(f"  df: {result2.df}")

result3 = scipy.differentiate.derivative(f, x, initial_step=0.5)
print(f"\ninitial_step=0.5 (valid):")
print(f"  success: {result3.success}")
print(f"  df: {result3.df}")
```

**Output:**
```
initial_step=0:
  success: False
  df: nan

initial_step=-0.5:
  success: False
  df: nan

initial_step=0.5 (valid):
  success: True
  df: 2.0
```

## Why This Is A Bug

1. The docstring describes `initial_step` as: "The (absolute) initial step size..." The word "(absolute)" strongly suggests this should be a positive value.

2. There is no input validation for `initial_step`. Looking at `_derivative_iv`, the validation only broadcasts it:
```python
initial_step = xp.asarray(initial_step)
temp = xp.broadcast_arrays(x, step_direction, initial_step)
x, step_direction, initial_step = temp
```

3. Invalid values cause silent failures rather than clear error messages, violating the principle of fail-fast input validation.

4. This creates an inconsistent API: `step_factor` is validated (albeit incorrectly as noted in a separate bug report), but `initial_step` is not validated at all.

## Fix

```diff
--- a/scipy/differentiate/_differentiate.py
+++ b/scipy/differentiate/_differentiate.py
@@ -48,6 +48,10 @@ def _derivative_iv(f, x, args, tolerances, maxiter, order, initial_step,
     step_direction = xp.asarray(step_direction)
     initial_step = xp.asarray(initial_step)
+
+    if xp.any(initial_step <= 0):
+        raise ValueError('`initial_step` must be positive.')
+
     temp = xp.broadcast_arrays(x, step_direction, initial_step)
     x, step_direction, initial_step = temp
```