# Bug Report: scipy.optimize Nelder-Mead Convergence Failure

**Target**: `scipy.optimize.minimize` (Nelder-Mead method)
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The Nelder-Mead optimization method converges to an incorrect minimum when started from certain initial points with extremely small coordinate values, yet reports `success=True`. For a simple convex quadratic function with a unique minimum at [1, -2], Nelder-Mead finds [1, ~0] starting from [0, ~1e-199].

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import numpy as np
from scipy.optimize import minimize

@given(st.floats(min_value=-3, max_value=3, allow_nan=False, allow_infinity=False),
       st.floats(min_value=-3, max_value=3, allow_nan=False, allow_infinity=False))
@settings(max_examples=100)
def test_minimize_methods_consistency(x0, y0):
    """Test that different methods find similar solutions for convex functions."""
    x_init = np.array([x0, y0])

    def convex_func(x):
        return (x[0] - 1) ** 2 + (x[1] + 2) ** 2

    methods = ['BFGS', 'Nelder-Mead', 'Powell']
    results = []

    for method in methods:
        try:
            result = minimize(convex_func, x_init, method=method)
            if result.success:
                results.append(result.x)
        except (ValueError, RuntimeError):
            pass

    if len(results) >= 2:
        for i in range(len(results) - 1):
            diff = np.linalg.norm(results[i] - results[i + 1])
            assert diff < 0.5, \
                f"Different methods should find similar solutions, difference: {diff}"
```

**Failing input**: `x0=0.0, y0=1.8004729751112097e-199`

## Reproducing the Bug

```python
import numpy as np
from scipy.optimize import minimize

x0 = np.array([0.0, 1.8e-199])

def func(x):
    return (x[0] - 1)**2 + (x[1] + 2)**2

result_nm = minimize(func, x0, method='Nelder-Mead')
result_bfgs = minimize(func, x0, method='BFGS')

print(f"True minimum: x=[1, -2], f=0")
print(f"\nNelder-Mead:")
print(f"  Solution: x={result_nm.x}")
print(f"  Function value: f={result_nm.fun}")
print(f"  Success: {result_nm.success}")
print(f"  Distance from true min: {np.linalg.norm(result_nm.x - [1, -2])}")

print(f"\nBFGS (for comparison):")
print(f"  Solution: x={result_bfgs.x}")
print(f"  Function value: f={result_bfgs.fun}")
print(f"  Distance from true min: {np.linalg.norm(result_bfgs.x - [1, -2])}")
```

Output:
```
True minimum: x=[1, -2], f=0

Nelder-Mead:
  Solution: x=[1.00000000e+000 -1.78246825e-197]
  Function value: f=4.0
  Success: True
  Distance from true min: 2.0

BFGS (for comparison):
  Solution: x=[0.99999998 -2.00000003]
  Function value: f=3.38e-16
  Distance from true min: 3.38e-08
```

## Why This Is A Bug

For the convex quadratic function f(x) = (x[0] - 1)² + (x[1] + 2)², there is a unique global minimum at x = [1, -2] with f(x) = 0.

Nelder-Mead converges to x ≈ [1, 0] with f(x) = 4, which is far from the true minimum. The issue appears to be triggered when the initial point has one coordinate that is extremely small (near machine epsilon or denormalized numbers like 1e-199).

The second coordinate value near 1e-199 seems to cause numerical issues in the Nelder-Mead simplex construction or convergence criteria, causing it to treat this dimension as effectively converged at ~0 rather than exploring towards -2.

This violates the expected behavior that:
1. Nelder-Mead should find the global minimum for simple convex functions
2. Different optimization methods should find similar solutions for smooth convex problems
3. The optimizer should not report success when it has clearly not found the optimum

## Fix

The issue likely stems from Nelder-Mead's convergence criteria or initial simplex construction not properly handling very small values. The fix would involve:

1. Reviewing the initial simplex construction in Nelder-Mead to handle coordinates near zero more robustly
2. Ensuring convergence criteria check absolute function value improvement, not just relative changes
3. Possibly adding adaptive scaling based on the magnitude of initial coordinates

A potential patch location would be in `/scipy/optimize/_optimize.py` in the `_minimize_neldermead` function, specifically in how the initial simplex is constructed when `x0` contains very small values.

Without access to the exact scipy source version being tested, a high-level fix would involve ensuring that when constructing the initial simplex, coordinates with very small absolute values (< 1e-100) are treated specially, perhaps by using an absolute step size rather than a relative one.