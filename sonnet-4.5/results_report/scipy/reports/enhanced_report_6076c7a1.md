# Bug Report: scipy.optimize.newton Accepts Invalid Negative rtol Values

**Target**: `scipy.optimize.newton`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `newton` function accepts negative `rtol` (relative tolerance) values without validation, while all other root-finding methods in `scipy.optimize` properly reject such values as mathematically invalid.

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

# Run the test
test_newton_rtol_validation()
```

<details>

<summary>
**Failing input**: `x0=1.0` (or any valid starting point)
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/33/hypo.py", line 25, in <module>
    test_newton_rtol_validation()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/33/hypo.py", line 6, in test_newton_rtol_validation
    st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/33/hypo.py", line 20, in test_newton_rtol_validation
    assert False, "newton should reject negative rtol"
           ^^^^^
AssertionError: newton should reject negative rtol
Falsifying example: test_newton_rtol_validation(
    x0=1.0,  # or any other generated value
)
```
</details>

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

<details>

<summary>
Inconsistent validation behavior between bisect and newton
</summary>
```
Testing bisect with negative rtol:
bisect rejected rtol=-0.1: rtol too small (-0.1 < 8.88178e-16)

Testing newton with negative rtol:
newton accepted rtol=-0.1, returned 2.0
```
</details>

## Why This Is A Bug

This violates expected behavior in three critical ways:

1. **Mathematical Incorrectness**: Relative tolerance is used in the convergence criterion `np.isclose(p, p0, rtol=rtol, atol=tol)` at line 342 of `/home/npc/.local/lib/python3.13/site-packages/scipy/optimize/_zeros_py.py`. The formula expands to `|a - b| <= atol + rtol * max(|a|, |b|)`. With negative `rtol`, this produces a negative threshold when `rtol * max(|a|, |b|)` exceeds `atol`, making the convergence check mathematically nonsensical.

2. **API Inconsistency**: All other root-finding methods in `scipy.optimize` validate `rtol`:
   - `bisect` (line 592-593): Requires `rtol >= 4*np.finfo(float).eps`
   - `ridder` (line 705-706): Requires `rtol >= 4*np.finfo(float).eps`
   - `brentq` (line 843-844): Requires `rtol >= 4*np.finfo(float).eps`
   - `brenth` (line 971-972): Requires `rtol >= 4*np.finfo(float).eps`
   - `newton` (line 109-394): **No validation of rtol at all**

3. **Silent Failure Risk**: Users could accidentally provide negative rtol (e.g., through typos or calculation errors) and the function would silently accept it, potentially leading to unexpected convergence behavior or non-termination in edge cases.

## Relevant Context

The `newton` function differs from other root-finding methods in that it has a default `rtol=0.0` (line 110), whereas others default to `4*np.finfo(float).eps`. This might explain why validation was overlooked - the function was designed to work primarily with `rtol=0` (absolute tolerance only).

However, since `newton` does accept and use the `rtol` parameter in its convergence check, it should validate that the value makes mathematical sense. The convergence check at line 342 relies on `np.isclose` which expects non-negative tolerance values.

Documentation reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.newton.html

Code location: `/home/npc/.local/lib/python3.13/site-packages/scipy/optimize/_zeros_py.py:109-394`

## Proposed Fix

```diff
--- a/scipy/optimize/_zeros_py.py
+++ b/scipy/optimize/_zeros_py.py
@@ -286,6 +286,8 @@ def newton(func, x0, fprime=None, args=(), tol=1.48e-8, maxiter=50,
     """
     if tol <= 0:
         raise ValueError(f"tol too small ({tol:g} <= 0)")
+    if rtol < 0:
+        raise ValueError(f"rtol must be non-negative ({rtol:g} < 0)")
     maxiter = operator.index(maxiter)
     if maxiter < 1:
         raise ValueError("maxiter must be greater than 0")
```