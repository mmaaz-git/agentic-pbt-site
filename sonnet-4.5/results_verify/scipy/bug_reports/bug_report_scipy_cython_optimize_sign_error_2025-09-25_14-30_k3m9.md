# Bug Report: scipy.optimize.cython_optimize Sign Error Detection Failure

**Target**: `scipy.optimize.cython_optimize._zeros.full_output_example`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `full_output_example` function fails to detect sign errors (no root bracketing) when the lower bound `xa` is extremely small (< ~1e-10) but still positive. The function incorrectly reports convergence (`error_num=0`) and returns `root=0.0` even though both `f(xa)` and `f(xb)` have the same sign.

## Property-Based Test

```python
from hypothesis import given, assume, strategies as st, settings
from scipy.optimize.cython_optimize._zeros import full_output_example
import math


def eval_polynomial(coeffs, x):
    a0, a1, a2, a3 = coeffs
    return a0 + a1*x + a2*x**2 + a3*x**3


@given(
    a0=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    a1=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    a2=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    a3=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    xa=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    xb=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=500)
def test_sign_error_when_no_sign_change(a0, a1, a2, a3, xa, xb):
    assume(xa < xb)
    assume(abs(xb - xa) > 1e-10)

    args = (a0, a1, a2, a3)
    f_xa = eval_polynomial(args, xa)
    f_xb = eval_polynomial(args, xb)

    assume(not math.isnan(f_xa) and not math.isinf(f_xa))
    assume(not math.isnan(f_xb) and not math.isinf(f_xb))
    assume(abs(f_xa) > 1e-100 and abs(f_xb) > 1e-100)

    if f_xa * f_xb > 0:
        xtol, rtol, mitr = 1e-6, 1e-6, 100
        result = full_output_example(args, xa, xb, xtol, rtol, mitr)

        assert result['error_num'] == -1, (
            f"Expected sign error (error_num=-1) when f(xa)*f(xb) > 0, "
            f"but got error_num={result['error_num']}. "
            f"f({xa})={f_xa}, f({xb})={f_xb}, Args: {args}"
        )
```

**Failing input**: `a0=0.0, a1=1.0, a2=0.0, a3=0.0, xa=9.398364172448502e-94, xb=1.0`

## Reproducing the Bug

```python
from scipy.optimize.cython_optimize._zeros import full_output_example


def eval_polynomial(coeffs, x):
    a0, a1, a2, a3 = coeffs
    return a0 + a1*x + a2*x**2 + a3*x**3


args = (0.0, 1.0, 0.0, 0.0)
xa = 1e-100
xb = 1.0

f_xa = eval_polynomial(args, xa)
f_xb = eval_polynomial(args, xb)

print(f'f(xa={xa}) = {f_xa}')
print(f'f(xb={xb}) = {f_xb}')
print(f'Both positive - no root bracketed')

result = full_output_example(args, xa, xb, 1e-6, 1e-6, 100)

print(f'error_num: {result["error_num"]} (expected -1 for sign error)')
print(f'root: {result["root"]}')

assert result['error_num'] == -1, f'Expected sign error but got error_num={result["error_num"]}'
```

Output:
```
f(xa=1e-100) = 1e-100
f(xb=1.0) = 1.0
Both positive - no root bracketed
error_num: 0 (expected -1 for sign error)
root: 0.0
AssertionError: Expected sign error but got error_num=0
```

## Why This Is A Bug

Bracketing root-finding methods (like brentq, which `full_output_example` uses internally) require that the function values at the bracket endpoints have opposite signs, i.e., `f(xa) * f(xb) < 0`. This ensures a root exists in the interval by the Intermediate Value Theorem.

The function's docstring states: "An error number of -1 means a sign error" - indicating that the function should detect and report when the bracketing condition is not met.

In this case:
- `f(xa) = 1e-100` (positive)
- `f(xb) = 1.0` (positive)
- `f(xa) * f(xb) > 0` (same sign)

However, the function returns `error_num=0` (indicating convergence) instead of the expected `error_num=-1` (sign error). The algorithm appears to treat very small values (< ~1e-10) as zero during the sign check, but this is incorrect when the value is actually positive.

This bug affects reliability: users expecting proper error detection may receive incorrect results without warning when dealing with functions that pass very close to (but don't cross) the x-axis.

## Fix

The root cause is likely in the sign-checking logic of the underlying C implementation. The sign check should use a proper comparison that doesn't treat very small non-zero values as zero, or should use a threshold that's documented and matches the function's contract.

A potential fix would be to modify the sign-checking code to:
1. Use exact sign comparison without tolerance: `if (f_xa > 0 && f_xb > 0) || (f_xa < 0 && f_xb < 0)`
2. Or document the threshold below which values are treated as zero
3. Or check `f_xa * f_xb > 0` directly without intermediate zero-equivalence checks

Without access to the C source code being modified, the exact fix location cannot be determined, but the issue is in the initial bracketing validation in the Brent-type algorithms in `scipy/optimize/Zeros/*.c` files.