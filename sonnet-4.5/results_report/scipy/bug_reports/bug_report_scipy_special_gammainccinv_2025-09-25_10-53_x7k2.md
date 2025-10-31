# Bug Report: scipy.special.gammainccinv Returns Incorrect Value for Small Shape Parameters

**Target**: `scipy.special.gammainccinv`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`scipy.special.gammainccinv(a, y)` returns 0.0 for very small shape parameters `a` (approximately `a < 0.0003`) when it should return a small positive value, violating the inverse property with `gammaincc`.

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
```

**Failing input**: `a = 1e-05, y = 0.5`

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

Output:
```
gammainccinv(1e-05, 0.5) = 0.0
gammaincc(1e-05, 0.0) = 1.0
Expected: 0.5
Error: 0.5
```

For comparison, when `a = 0.001` (which works correctly):
```
gammainccinv(0.001, 0.5) = 5.244206408274974e-302
gammaincc(0.001, 5.244206408274974e-302) = 0.5
Error: ~1e-16
```

## Why This Is A Bug

`gammainccinv(a, y)` is documented as the inverse of the regularized upper incomplete gamma function `gammaincc`. For any valid inputs, the property `gammaincc(a, gammainccinv(a, y)) ≈ y` should hold.

For very small `a` (< ~0.0003), `gammainccinv` returns exactly 0.0, which causes `gammaincc(a, 0) = 1.0`, severely violating the inverse property. The function should instead return a very small positive number (as it correctly does for `a = 0.001`).

This suggests that the implementation has a threshold or underflow issue where it incorrectly returns 0.0 instead of computing the proper (very small) result.

## Fix

The bug likely occurs in the numerical computation of `gammainccinv` when the shape parameter `a` is very small. The function should handle very small `a` values more carefully, possibly using asymptotic expansions or higher precision arithmetic to avoid underflow to 0.0.

A potential fix would involve:
1. Detecting when `a` is very small (e.g., `a < 0.001`)
2. Using a specialized numerical method or asymptotic formula for small `a`
3. Ensuring the result is never exactly 0.0 when `y` is in the valid range `(0, 1)`

Without access to the source code, a specific patch cannot be provided, but the fix should ensure that for small `a`, the function returns appropriately small positive values that maintain the inverse relationship with `gammaincc`.