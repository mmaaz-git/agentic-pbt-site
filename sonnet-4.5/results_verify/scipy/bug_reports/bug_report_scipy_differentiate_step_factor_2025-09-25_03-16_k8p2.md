# Bug Report: scipy.differentiate.derivative Allows step_factor=0

**Target**: `scipy.differentiate.derivative`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `derivative` function's input validation accepts `step_factor=0`, but this value causes division by zero during computation. The validation at lines 31-33 only checks `step_factor >= 0`, when it should require `step_factor > 0`.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from scipy.differentiate import derivative
import numpy as np
import pytest

@given(st.just(0.0))
def test_derivative_should_reject_zero_step_factor(step_factor):
    with pytest.raises(ValueError, match="step_factor"):
        derivative(np.sin, 1.0, step_factor=step_factor)
```

**Failing input**: `step_factor=0.0`

## Reproducing the Bug

```python
import numpy as np
from scipy.differentiate import derivative

result = derivative(np.sin, 1.0, step_factor=0.0)
```

Expected: `ValueError` during input validation stating that `step_factor` must be positive
Actual: The function accepts `step_factor=0.0` and later encounters division by zero or produces invalid results

## Why This Is A Bug

The validation logic at lines 31-33 checks:
```python
if (not np.issubdtype(tols.dtype, np.number) or np.any(tols < 0)
        or np.any(np.isnan(tols)) or tols.shape != (3,)):
    raise ValueError(message)
```

This accepts `step_factor=0` since `0 >= 0`. However, at line 468 in `pre_func_eval`:
```python
hc = h / c**xp.arange(n, dtype=work.dtype)
```

When `c` (which equals `step_factor`) is 0, this causes division by zero: `c**0 = 1`, `c**1 = 0`, leading to `h / 0`.

The documentation at lines 109-114 states that `step_factor < 1` is allowed for specific use cases, but `step_factor = 0` has no valid mathematical interpretation as it would mean the step size is reduced by a factor of 0.

## Fix

```diff
diff --git a/scipy/differentiate/_differentiate.py b/scipy/differentiate/_differentiate.py
index 1234567..abcdefg 100644
--- a/scipy/differentiate/_differentiate.py
+++ b/scipy/differentiate/_differentiate.py
@@ -27,9 +27,9 @@ def _derivative_iv(f, x, args, tolerances, maxiter, order, initial_step,
-    message = 'Tolerances and step parameters must be non-negative scalars.'
+    message = 'Tolerances must be non-negative scalars and step_factor must be positive.'
     tols = np.asarray([atol if atol is not None else 1,
                        rtol if rtol is not None else 1,
                        step_factor])
-    if (not np.issubdtype(tols.dtype, np.number) or np.any(tols < 0)
+    if (not np.issubdtype(tols.dtype, np.number) or np.any(tols[:2] < 0) or tols[2] <= 0
             or np.any(np.isnan(tols)) or tols.shape != (3,)):
         raise ValueError(message)
```