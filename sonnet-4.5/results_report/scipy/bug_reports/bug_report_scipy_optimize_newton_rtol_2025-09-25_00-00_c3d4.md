# Bug Report: scipy.optimize.newton Does Not Validate rtol Parameter

**Target**: `scipy.optimize.newton`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `newton` function does not validate its `rtol` parameter, allowing negative and potentially nonsensical values. This is inconsistent with the other root-finding methods (`bisect`, `ridder`, `brenth`, `brentq`) which all validate that `rtol >= 4 * np.finfo(float).eps`.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume, settings
from scipy.optimize import newton


@given(
    st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=300)
def test_newton_rtol_validation(x0):
    assume(abs(x0) > 0.1)

    def f(x):
        return x**2 - 4

    def fprime(x):
        return 2 * x

    try:
        root = newton(f, x0, fprime=fprime, rtol=-0.1, disp=False)
        assert False, "newton should reject negative rtol"
    except ValueError:
        pass
```

**Failing input**: Any `rtol < 0` value, e.g., `rtol=-0.1`

## Reproducing the Bug

```python
from scipy.optimize import newton, bisect

def f(x):
    return x**2 - 4

def fprime(x):
    return 2 * x

print("Testing bisect with negative rtol:")
try:
    root = bisect(f, 0.0, 3.0, rtol=-0.1, disp=False)
    print(f"bisect accepted rtol=-0.1")
except ValueError as e:
    print(f"bisect rejected rtol=-0.1: {e}")

print("\nTesting newton with negative rtol:")
try:
    root = newton(f, 1.0, fprime=fprime, rtol=-0.1, disp=False)
    print(f"newton accepted rtol=-0.1, returned {root}")
except ValueError as e:
    print(f"newton rejected rtol=-0.1: {e}")
```

Output:
```
Testing bisect with negative rtol:
bisect rejected rtol=-0.1: rtol too small (-0.1 < 8.88178e-16)

Testing newton with negative rtol:
newton accepted rtol=-0.1, returned 2.0
```

## Why This Is A Bug

1. **API Inconsistency**: All other root-finding methods in `scipy.optimize` validate `rtol` to ensure it's at least `4 * np.finfo(float).eps` (approximately 8.88e-16). The `newton` function should have the same validation for consistency.

2. **Undefined Behavior**: A negative relative tolerance makes no mathematical sense. The convergence criterion `np.isclose(p, p0, rtol=rtol, atol=tol)` uses `rtol` in the formula `|a - b| <= atol + rtol * max(|a|, |b|)`. A negative `rtol` would make this check behave incorrectly.

3. **User Error Prevention**: Without validation, user typos or bugs (like accidentally negating `rtol`) go undetected, potentially causing incorrect results or non-convergence.

## Fix

Add validation in the `newton` function similar to other root-finding methods:

```diff
def newton(func, x0, fprime=None, args=(), tol=1.48e-8, maxiter=50,
           fprime2=None, x1=None, rtol=0.0,
           full_output=False, disp=True):
    if tol <= 0:
        raise ValueError(f"tol too small ({tol:g} <= 0)")
+   _rtol = 4 * np.finfo(float).eps
+   if rtol < 0:
+       raise ValueError(f"rtol must be non-negative, got {rtol:g}")
    maxiter = operator.index(maxiter)
    if maxiter < 1:
        raise ValueError("maxiter must be greater than 0")
```

Note: Unlike `bisect`/`ridder`/`brenth`/`brentq`, `newton` has a default `rtol=0.0`, so the validation should only check for negative values, not enforce a minimum positive value.