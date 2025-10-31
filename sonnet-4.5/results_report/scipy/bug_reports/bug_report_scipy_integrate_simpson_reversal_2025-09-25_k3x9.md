# Bug Report: scipy.integrate.simpson Reversal Property Violation

**Target**: `scipy.integrate.simpson`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `simpson` function violates the fundamental integral reversal property (∫ₐᵇ f(x)dx = -∫ᵇₐ f(x)dx) when given an even number of sample points. This is a mathematical property that should hold for any numerical integration method.

## Property-Based Test

```python
import numpy as np
from scipy import integrate
from hypothesis import given, strategies as st, settings

@given(
    n=st.sampled_from([4, 6, 8, 10]),
    a=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
    b=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=200)
def test_simpson_reversal_even_n(n, a, b):
    from hypothesis import assume
    assume(b > a)
    assume(abs(b - a) > 1e-6)

    x_forward = np.linspace(a, b, n)
    y = np.random.randn(n)

    x_backward = x_forward[::-1]
    y_backward = y[::-1]

    result_forward = integrate.simpson(y, x=x_forward)
    result_backward = integrate.simpson(y_backward, x=x_backward)

    assert np.isclose(result_forward, -result_backward, rtol=1e-10, atol=1e-10)
```

**Failing input**: `n=4, a=0.99999, b=54.645, y=random`
This test consistently fails for even values of n (4, 6, 8, 10) but passes for odd values of n (3, 5, 7, 9, 11).

## Reproducing the Bug

```python
import numpy as np
from scipy import integrate

n = 4
a, b = 0.0, 10.0
x_forward = np.linspace(a, b, n)
y = x_forward ** 2

x_backward = x_forward[::-1]
y_backward = y[::-1]

result_forward = integrate.simpson(y, x=x_forward)
result_backward = integrate.simpson(y_backward, x=x_backward)

print(f"result_forward (∫_0^10 x² dx):  {result_forward}")
print(f"result_backward (∫_10^0 x² dx): {result_backward}")
print(f"Sum (should be ~0): {result_forward + result_backward}")
```

**Output**:
```
result_forward (∫_0^10 x² dx):  333.3333333333333
result_backward (∫_10^0 x² dx): -335.55555555555554
Sum (should be ~0): -2.2222222222222214
```

## Why This Is A Bug

The mathematical property ∫ₐᵇ f(x)dx = -∫ᵇₐ f(x)dx is fundamental to integration and must hold for any numerical integration method. When we reverse both the x-coordinates and y-values, we're computing the integral in the opposite direction, which should yield the negative of the original result.

The `trapezoid` function in scipy.integrate correctly implements this property (as documented in its docstring at line 90-93), but `simpson` does not when n is even.

The root cause is in the special correction formula used for even n (lines 463-534 in `_quadrature.py`). When n is even, Simpson's rule uses a different formula for the last interval that is not symmetric under reversal.

## Fix

The even-n correction formula needs to be modified to preserve the reversal property. Specifically, the asymmetric treatment of the last interval (lines 478-532) should be made symmetric, or alternatively, the function should apply the same correction symmetrically from both ends when n is even.

A potential fix would be to ensure that the correction for the last interval in the forward direction is the exact negative of the correction for the first interval in the backward direction.

Alternatively, the function could document this as a known limitation and recommend using an odd number of points for applications where the reversal property is important.