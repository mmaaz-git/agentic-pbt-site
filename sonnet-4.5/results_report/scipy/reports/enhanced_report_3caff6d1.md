# Bug Report: scipy.optimize.cython_optimize Sign Error Detection Failure for Small Positive Values

**Target**: `scipy.optimize.cython_optimize._zeros.full_output_example`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `full_output_example` function incorrectly returns successful convergence (error_num=0) instead of reporting a sign error (error_num=-1) when both function values have the same sign but one value is extremely small (< ~1e-50).

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


if __name__ == "__main__":
    test_sign_error_when_no_sign_change()
```

<details>

<summary>
**Failing input**: `a0=0.0, a1=10.0, a2=0.0, a3=0.0, xa=1.0149495420240701e-101, xb=1.0`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/56/hypo.py", line 44, in <module>
    test_sign_error_when_no_sign_change()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/56/hypo.py", line 12, in test_sign_error_when_no_sign_change
    a0=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/56/hypo.py", line 36, in test_sign_error_when_no_sign_change
    assert result['error_num'] == -1, (
           ^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Expected sign error (error_num=-1) when f(xa)*f(xb) > 0, but got error_num=0. f(1.0149495420240701e-101)=1.0149495420240701e-100, f(1.0)=10.0, Args: (0.0, 10.0, 0.0, 0.0)
Falsifying example: test_sign_error_when_no_sign_change(
    # The test sometimes passed when commented parts were varied together.
    a0=0.0,
    a1=10.0,  # or any other generated value
    a2=0.0,
    a3=0.0,
    xa=1.0149495420240701e-101,
    xb=1.0,  # or any other generated value
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/worker_/56/hypo.py:37
```
</details>

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
print()

result = full_output_example(args, xa, xb, 1e-6, 1e-6, 100)

print(f'error_num: {result["error_num"]} (expected -1 for sign error)')
print(f'root: {result["root"]}')
print()

assert result['error_num'] == -1, f'Expected sign error but got error_num={result["error_num"]}'
```

<details>

<summary>
AssertionError: Expected sign error but got error_num=0
</summary>
```
f(xa=1e-100) = 1e-100
f(xb=1.0) = 1.0
Both positive - no root bracketed

error_num: 0 (expected -1 for sign error)
root: 0.0

Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/56/repo.py", line 27, in <module>
    assert result['error_num'] == -1, f'Expected sign error but got error_num={result["error_num"]}'
           ^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Expected sign error but got error_num=0
```
</details>

## Why This Is A Bug

This violates the fundamental mathematical requirement of bracketing root-finding methods. The function claims to find a root using a Brent-type bracketing algorithm, which requires that f(xa) and f(xb) have opposite signs to guarantee a root exists in [xa, xb] by the Intermediate Value Theorem.

The scipy codebase explicitly defines error codes in `scipy/optimize/_zeros_py.py`:
- `_ECONVERGED = 0` means "converged"
- `_ESIGNERR = -1` means "sign error"
- These constants "Must agree with CONVERGED, SIGNERR, CONVERR, ... in zeros.h"

When f(xa) = 1e-100 and f(xb) = 1.0, both values are positive (same sign). The function should detect this condition and return `error_num = -1` to indicate a sign error. Instead, it returns `error_num = 0` (converged) with `root = 0.0`, which is mathematically incorrect and misleading to users.

The bug appears when function values are smaller than approximately 1e-50. These values are well within the valid range of float64 (minimum positive value is ~4.94e-324), so they should not be treated as zero. This incorrect behavior can cause silent failures in scientific computations where small but non-zero values are meaningful.

## Relevant Context

The function's docstring states it returns "the zero function error number" but doesn't specify what each error code means. However, the error codes are clearly defined in the scipy optimization module.

The scipy.optimize.brentq documentation states: "f must be continuous. f(a) and f(b) must have opposite signs." This is a standard requirement for all bracketing methods that `full_output_example` uses internally.

Testing shows the bug threshold is around 1e-50:
- When f(xa) > 1e-50: correctly returns error_num = -1
- When f(xa) < 1e-50: incorrectly returns error_num = 0

This suggests the underlying C implementation may be using an undocumented tolerance that incorrectly treats small positive values as zero during sign checking.

## Proposed Fix

The sign check logic in the underlying C implementation needs to be fixed to properly detect when both function values have the same sign, regardless of magnitude. Without access to modify the Cython/C source directly, here's a high-level fix approach:

The issue is likely in the initial bracket validation in the Brent algorithm implementation. The fix should:

1. Check the actual signs of f(xa) and f(xb) without treating small values as zero
2. Return SIGNERR (-1) when the signs are the same
3. Only proceed with root finding when signs are genuinely opposite

The sign check should use one of these approaches:
- Direct multiplication check: `if (f_xa * f_xb > 0) return SIGNERR;`
- Explicit sign comparison: `if ((f_xa > 0 && f_xb > 0) || (f_xa < 0 && f_xb < 0)) return SIGNERR;`
- If a zero tolerance is necessary, it should be documented and use a reasonable threshold (e.g., machine epsilon rather than 1e-50)