# Bug Report: scipy.special.gammainccinv Returns Zero for Small Shape Parameters

**Target**: `scipy.special.gammainccinv`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`scipy.special.gammainccinv(a, y)` incorrectly returns 0.0 for shape parameters a < 0.0005, violating the documented inverse relationship with `gammaincc`.

## Property-Based Test

```python
import scipy.special as sp
from hypothesis import given, strategies as st


@given(st.floats(min_value=1e-10, max_value=1e-02, allow_nan=False, allow_infinity=False))
def test_gammainccinv_inverse_property(a):
    y = 0.5
    x = sp.gammainccinv(a, y)
    result = sp.gammaincc(a, x)
    assert abs(result - y) < 1e-6, \
        f"For a={a}, gammainccinv({a}, {y}) = {x}, but gammaincc({a}, {x}) = {result}, expected {y}"

if __name__ == "__main__":
    test_gammainccinv_inverse_property()
```

<details>

<summary>
**Failing input**: `a=5.338506319915343e-05`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/2/hypo.py", line 14, in <module>
    test_gammainccinv_inverse_property()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/2/hypo.py", line 6, in test_gammainccinv_inverse_property
    def test_gammainccinv_inverse_property(a):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/2/hypo.py", line 10, in test_gammainccinv_inverse_property
    assert abs(result - y) < 1e-6, \
           ^^^^^^^^^^^^^^^^^^^^^^
AssertionError: For a=5.338506319915343e-05, gammainccinv(5.338506319915343e-05, 0.5) = 0.0, but gammaincc(5.338506319915343e-05, 0.0) = 1.0, expected 0.5
Falsifying example: test_gammainccinv_inverse_property(
    a=5.338506319915343e-05,
)
```
</details>

## Reproducing the Bug

```python
import scipy.special as sp

a = 1e-05
y = 0.5

x = sp.gammainccinv(a, y)
print(f"gammainccinv({a}, {y}) = {x}")

result = sp.gammaincc(a, x)
print(f"gammaincc({a}, {x}) = {result}")
print(f"Expected: {y}")
print(f"Error: {abs(result - y)}")
```

<details>

<summary>
Output showing incorrect inverse computation
</summary>
```
gammainccinv(1e-05, 0.5) = 0.0
gammaincc(1e-05, 0.0) = 1.0
Expected: 0.5
Error: 0.5
```
</details>

## Why This Is A Bug

The documentation for `gammainccinv` explicitly states it is the inverse of the regularized upper incomplete gamma function `gammaincc`. The function's docstring includes the example: `sc.gammaincc(a, sc.gammainccinv(a, x))` should return the original `x` values.

Testing reveals a clear threshold bug:
- For `a = 0.001`: Works correctly, returns `5.244e-302` (extremely small but non-zero)
- For `a = 0.0005`: Fails, returns exactly `0.0`
- For all `a < 0.0005`: Consistently returns `0.0` instead of the correct small positive value

When `gammainccinv` returns 0.0, calling `gammaincc(a, 0.0)` always returns 1.0 regardless of the original y value. This violates the fundamental mathematical property that these functions should be inverses of each other for all valid inputs (a > 0, 0 ≤ y ≤ 1).

The function successfully computes extremely small values (~10^-302) for a = 0.001, proving this isn't a floating-point limitation but rather an implementation bug with a hardcoded threshold or premature underflow to zero.

## Relevant Context

The NIST Digital Library reference (https://dlmf.nist.gov/8.2#E4) defines the regularized upper incomplete gamma function mathematically for all positive a values. There is no mathematical discontinuity at a = 0.0005 that would justify this behavior.

The scipy documentation shows that `gammainccinv` should handle the full range from 0 to 1 for the y parameter and positive values for a, with no mentioned restrictions on minimum a values.

Testing shows the threshold is precisely between `a = 0.001` (works) and `a = 0.0005` (fails), suggesting a hardcoded threshold in the implementation rather than a numerical precision issue.

## Proposed Fix

The implementation likely contains a threshold check or early return that incorrectly handles small shape parameters. A high-level fix would involve:

1. Identifying where the implementation prematurely returns 0.0 for small a values
2. Using appropriate asymptotic expansions for small a to compute the correct result
3. Ensuring the function never returns exactly 0.0 when y is in (0, 1), as the mathematically correct result is always positive

Without access to the C/Fortran implementation, a specific patch cannot be provided, but the fix should remove any artificial threshold that causes the function to return 0.0 for small but valid shape parameters.