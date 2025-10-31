# Bug Report: numpy.polynomial Power Operator Inconsistency

**Target**: `numpy.polynomial.Polynomial` and `numpy.polynomial.Chebyshev`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `**` operator produces different coefficient arrays than repeated `*` operations when polynomial multiplication results in numerical underflow. This violates the mathematical identity that `p**n` should equal `p*p*...*p` (n times).

## Property-Based Test

```python
import numpy as np
import numpy.polynomial as poly
from hypothesis import given, settings, strategies as st


@settings(max_examples=500)
@given(
    st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e3, max_value=1e3), min_size=1, max_size=5),
    st.integers(min_value=2, max_value=3)
)
def test_power_by_repeated_multiplication(coeffs, n):
    p = poly.Polynomial(coeffs)

    result_power = p ** n
    result_mult = p
    for _ in range(n - 1):
        result_mult = result_mult * p

    np.testing.assert_array_equal(result_power.coef, result_mult.coef)
```

**Failing input**: `coeffs=[0.0, 1.1125369292536007e-308], n=2`

## Reproducing the Bug

```python
import numpy as np
import numpy.polynomial as poly

p = poly.Polynomial([0.0, 1e-308])

result_power = p ** 2
result_mult = p * p

print(f'p**2 coefficients: {result_power.coef}')
print(f'p*p coefficients: {result_mult.coef}')
print(f'Arrays equal: {np.array_equal(result_power.coef, result_mult.coef)}')
```

Output:
```
p**2 coefficients: [0. 0. 0.]
p*p coefficients: [0.]
Arrays equal: False
```

## Why This Is A Bug

The `**` operator should be mathematically equivalent to repeated multiplication. When the coefficient `1e-308` is squared, it underflows to `0.0`. The `*` operator correctly trims the resulting zero coefficients, producing `[0.]`. However, the `**` operator preserves the polynomial structure without trimming, producing `[0., 0., 0.]`.

This inconsistency violates the fundamental algebraic property that `p**n` should equal `p*p*...*p`. While both results are mathematically equivalent to the zero polynomial, the coefficient representations differ, which can cause issues in code that compares polynomial objects or relies on consistent array shapes.

**Scope**:
- Affects: `Polynomial` and `Chebyshev` classes for all `n >= 2`
- Does not affect: `Legendre`, `Hermite`, `HermiteE`, `Laguerre` classes

## Fix

The `__pow__` method should apply the same coefficient trimming as the `__mul__` method. After computing the result of exponentiation, the method should call `.trim()` on the result before returning it, ensuring consistent behavior with repeated multiplication.

The fix would involve modifying the `__pow__` implementation in the base class or the affected polynomial classes to include:

```diff
def __pow__(self, other):
    # ... existing exponentiation logic ...
-   return result
+   return result.trim()
```

Alternatively, investigate why `__mul__` trims but `__pow__` doesn't, and ensure both follow the same code path for trimming zero coefficients.