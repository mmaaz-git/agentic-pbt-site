# Bug Report: scipy.optimize.least_squares False Convergence with Small Initial Values

**Target**: `scipy.optimize.least_squares`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`scipy.optimize.least_squares` with method='lm' (Levenberg-Marquardt) reports `success=True` but fails to converge when given initial values that are very small or zero. The gradient norm remains large (>2.0) even though the function claims successful termination via `ftol` condition.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
import numpy as np
from scipy.optimize import least_squares

@given(
    st.lists(st.floats(min_value=-3, max_value=3, allow_nan=False, allow_infinity=False),
             min_size=3, max_size=6)
)
def test_least_squares_gradient_zero_at_optimum(x0_list):
    """Test that at optimum, gradient J^T r is near zero."""
    x0 = np.array(x0_list)
    assume(np.linalg.norm(x0) < 10)
    n = len(x0)

    np.random.seed(hash(tuple(x0_list)) % (2**32))
    A = np.random.randn(n + 2, n)
    b = np.random.randn(n + 2)

    def residual_func(x):
        return A @ x - b

    def jacobian_func(x):
        return A

    result = least_squares(residual_func, x0, jac=jacobian_func, method='lm')

    if result.success:
        final_residual = residual_func(result.x)
        jac = jacobian_func(result.x)
        gradient = jac.T @ final_residual
        gradient_norm = np.linalg.norm(gradient)

        assert gradient_norm < 1e-6, \
            f"Gradient J^T r should be near zero at optimum, got norm {gradient_norm}"
```

**Failing input**: `x0_list = [0.0, 0.0, 1.2751402521491925e-80]`

## Reproducing the Bug

```python
import numpy as np
from scipy.optimize import least_squares

x0 = np.array([0.0, 0.0, 1.2751402521491925e-80])
n = len(x0)

np.random.seed(hash(tuple([0.0, 0.0, 1.2751402521491925e-80])) % (2**32))
A = np.random.randn(n + 2, n)
b = np.random.randn(n + 2)

def residual_func(x):
    return A @ x - b

def jacobian_func(x):
    return A

result = least_squares(residual_func, x0, jac=jacobian_func, method='lm')

print(f"success: {result.success}")
print(f"message: {result.message}")
print(f"x: {result.x}")
print(f"optimality: {result.optimality}")

gradient = jacobian_func(result.x).T @ residual_func(result.x)
gradient_norm = np.linalg.norm(gradient)
print(f"Gradient norm: {gradient_norm}")

x0_better = np.ones(n)
result2 = least_squares(residual_func, x0_better, jac=jacobian_func, method='lm')
gradient2_norm = np.linalg.norm(jacobian_func(result2.x).T @ residual_func(result2.x))

print(f"\nWith better initial guess:")
print(f"success: {result2.success}")
print(f"Gradient norm: {gradient2_norm}")
```

Output:
```
success: True
message: `ftol` termination condition is satisfied.
x: [0.00000000e+00 0.00000000e+00 1.27514025e-80]
optimality: 2.050686125311085
Gradient norm: 2.163824636993796

With better initial guess:
success: True
Gradient norm: 8.481535939042889e-16
```

## Why This Is A Bug

For least squares optimization, at the true optimum, the gradient J^T r (where J is the Jacobian and r is the residual) must be zero (or near machine precision). This is a fundamental mathematical property of least squares.

In this case:
1. The function reports `success=True` indicating it found an optimal solution
2. The gradient norm is 2.16, which is far from zero
3. The solution vector didn't change from the initial guess at all
4. The `optimality` field correctly shows 2.05 (non-zero), contradicting the success status

When starting from a better initial guess (ones), the same problem converges correctly with gradient norm ~1e-15.

This indicates the Levenberg-Marquardt implementation is incorrectly terminating based on the `ftol` condition without properly checking gradient convergence when the initial values are very small or zero.

## Fix

The bug appears to be in the termination logic for the Levenberg-Marquardt method. The `ftol` condition checks for small changes in the cost function, but when starting from values near zero, the cost function may appear stable even though the gradient is large.

Suggested fix: The success criteria should require BOTH:
1. Cost function convergence (`ftol` condition), AND
2. Gradient convergence (`gtol` condition)

Or at minimum, when `ftol` triggers termination, verify that the gradient norm is below a reasonable threshold before reporting `success=True`.

A patch would likely be in `/scipy/optimize/_lsq/levenberg_marquardt.py` or similar, adding an additional gradient check in the termination conditions.