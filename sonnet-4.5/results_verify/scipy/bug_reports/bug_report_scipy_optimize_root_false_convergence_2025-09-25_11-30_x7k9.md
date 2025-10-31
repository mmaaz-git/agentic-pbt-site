# Bug Report: scipy.optimize.root False Convergence

**Target**: `scipy.optimize.root`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`scipy.optimize.root()` with the default 'hybr' method (and 'lm' method) incorrectly reports `success=True` when residuals are large (far from zero), violating its documented purpose to "find a root of a vector function."

## Property-Based Test

```python
import numpy as np
import scipy.optimize
from hypothesis import given, strategies as st
from hypothesis.extra import numpy as npst
import pytest


@given(
    npst.arrays(
        dtype=np.float64,
        shape=3,
        elements=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False)
    ),
)
def test_root_multivariate_convergence(x0):
    """
    Property: root() should find solutions where f(x) ≈ 0 when success=True.

    Evidence: root docstring states "Find a root of a vector function"
    """
    def func(x):
        return [
            x[0] + 0.5 * (x[0] - x[1])**3 - 1,
            0.5 * (x[1] - x[0])**3 + x[1],
            x[2]**2 - 1
        ]

    result = scipy.optimize.root(func, x0)

    if result.success:
        f_values = func(result.x)
        for i, f_val in enumerate(f_values):
            assert abs(f_val) < 1e-5, \
                f"Equation {i}: f(x) = {f_val} should be close to 0"
```

**Failing input**: `x0 = array([0.0, 1.4e-11, 0.0])`

## Reproducing the Bug

```python
import numpy as np
import scipy.optimize

x0 = np.array([0.0, 1.4e-11, 0.0])

def equations(x):
    return [
        x[0] + 0.5 * (x[0] - x[1])**3 - 1,
        0.5 * (x[1] - x[0])**3 + x[1],
        x[2]**2 - 1
    ]

result = scipy.optimize.root(equations, x0, method='hybr')

print(f"success: {result.success}")
print(f"message: {result.message}")
print(f"residuals: {result.fun}")
print(f"max|residual|: {max(abs(r) for r in result.fun)}")
```

**Output:**
```
success: True
message: The solution converged.
residuals: [-9.99999979e-01  1.41058181e-11  8.11219663e-03]
max|residual|: 0.99999997939638
```

## Why This Is A Bug

The function claims "The solution converged" with `success=True`, but the maximum residual is approximately 1.0, meaning the first equation is not satisfied at all. According to the docstring, `root()` should "find a root of a vector function," which means finding `x` such that `f(x) ≈ 0`. Users relying on the `success` flag would incorrectly believe they have found a valid root.

**Additional evidence:**
1. Setting explicit `tol=1e-10` correctly reports `success=False` for this case
2. Alternative methods ('broyden1', 'broyden2') correctly find roots with small residuals
3. The 'lm' method exhibits the same bug

The issue appears to be that the default 'hybr' and 'lm' methods use step-size-based convergence criteria rather than residual-based criteria, and report success when step sizes are small even if residuals remain large.

## Fix

The fix should ensure that `success=True` is only reported when residuals are actually small. One approach:

```diff
--- a/scipy/optimize/_root.py
+++ b/scipy/optimize/_root.py
@@ -hybr_method_handler
+    # Check residuals before declaring success
+    if info == 1:
+        max_residual = np.max(np.abs(fvec))
+        if max_residual > (tol if tol is not None else 1.49012e-08):
+            info = 5  # Change to "not converged" status
+
     sol = OptimizeResult(x=x, success=(info == 1), status=info)
```

Alternatively, improve documentation to clarify that `success=True` means "step size converged" not "residuals are small," though this would be less user-friendly.