# Bug Report: scipy.optimize.ridder Convergence Failure with Custom Tolerances

**Target**: `scipy.optimize.ridder`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `ridder` root-finding method fails to converge when using custom tolerance settings (`xtol`, `rtol`) for certain inputs, even though it finds the root to machine precision. The bug exhibits asymmetric behavior: it fails to converge when finding negative roots but succeeds for positive roots with the same tolerance settings.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st, assume
from scipy.optimize import ridder

@given(
    st.floats(min_value=-10, max_value=-0.1, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0.1, max_value=10, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_ridder_converges_with_custom_tolerance(a, b):
    assume(abs(a - b) > 1e-6)

    def f(x):
        return x * x - 2.0

    fa, fb = f(a), f(b)
    assume(fa * fb < 0)

    result = ridder(f, a, b, xtol=1e-3, rtol=1e-3, full_output=True, disp=False)
    root, info = result

    assert info.converged, f"ridder failed to converge for interval [{a}, {b}]"
    assert abs(f(root)) < 1e-6, f"f(root) = {f(root)}, expected ~0"
```

**Failing input**: `a=-2.0, b=1.0` (or any interval containing the negative root -√2)

## Reproducing the Bug

```python
from scipy.optimize import ridder

def f(x):
    return x * x - 2.0

print("Test case: Finding root of f(x) = x² - 2 in [-2, 1]")
print("Expected root: -√2 ≈ -1.41421356...")

result = ridder(f, -2.0, 1.0, xtol=1e-3, rtol=1e-3, full_output=True, disp=False)
root, info = result

print(f"Converged: {info.converged}")
print(f"Iterations: {info.iterations}")
print(f"Root: {root:.15f}")
print(f"f(root): {f(root):.2e}")

print("\nComparison: Same function, interval [0, 2] (positive root)")
result2 = ridder(f, 0.0, 2.0, xtol=1e-3, rtol=1e-3, full_output=True)
root2, info2 = result2
print(f"Converged: {info2.converged}")
print(f"Iterations: {info2.iterations}")
print(f"Root: {root2:.15f}")
print(f"f(root): {f(root2):.2e}")
```

**Output:**
```
Test case: Finding root of f(x) = x² - 2 in [-2, 1]
Expected root: -√2 ≈ -1.41421356...
Converged: False
Iterations: 100
Root: -1.414213562373095
f(root): -4.44e-16

Comparison: Same function, interval [0, 2] (positive root)
Converged: True
Iterations: 3
Root: 1.413193860008073
f(root): -2.88e-03
```

## Why This Is A Bug

The `ridder` method finds the negative root to **machine precision** (f(root) = -4.44e-16, which is essentially zero), yet reports `converged: False` and exhausts all 100 iterations. Meanwhile, for the positive root in the same function, it converges successfully in just 3 iterations with much lower precision (f(root) = -2.88e-03).

This violates the documented behavior: the docstring claims the computed root will satisfy the tolerance criterion `abs(x - x0) <= xtol + rtol * abs(x0)`, but the algorithm fails to recognize when this criterion is met.

The asymmetry between positive and negative roots suggests a bug in the C implementation of the convergence check in `_zeros._ridder`, possibly related to:
1. Sign handling in the convergence criterion
2. Incorrect bracketing interval calculations
3. Issues with `rtol * abs(x0)` when x0 is negative

This bug affects users who need custom tolerances for their specific applications, causing unnecessary failures despite finding correct roots.

## Fix

The bug is in the C implementation (`scipy/optimize/Zeros/ridder.c`), which is not directly accessible from Python. The convergence criterion implementation needs to be reviewed for correct handling of negative roots.

A workaround for users is to use default tolerances or other root-finding methods (brentq, brenth) which do not exhibit this issue:

```python
from scipy.optimize import brentq

root = brentq(f, -2.0, 1.0, xtol=1e-3, rtol=1e-3)
```