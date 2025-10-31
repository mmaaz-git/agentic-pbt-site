# Bug Report: scipy.integrate.simpson Reversal Property Violation

**Target**: `scipy.integrate.simpson`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `simpson` function violates the fundamental mathematical property that reversing integration limits should negate the result: ∫ᵇₐ f(x)dx = -∫ₐᵇ f(x)dx. This occurs when the number of sample points is even (N % 2 == 0).

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import numpy as np
from scipy.integrate import simpson

@given(
    st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
             min_size=3, max_size=100)
)
@settings(max_examples=500)
def test_simpson_reverse_negates(y):
    n = len(y)
    y_arr = np.array(y)
    x_sorted = np.arange(n, dtype=float)

    result_forward = simpson(y_arr, x=x_sorted)
    result_backward = simpson(y_arr[::-1], x=x_sorted[::-1])

    assert np.isclose(result_forward, -result_backward, rtol=1e-10, atol=1e-10)
```

**Failing input**: `y = [0.0, 0.0, 0.0, 1.0]`

## Reproducing the Bug

```python
from scipy.integrate import simpson
import numpy as np

y = np.array([0.0, 0.0, 0.0, 1.0])
x = np.array([0.0, 1.0, 2.0, 3.0])

forward = simpson(y, x=x)
backward = simpson(y[::-1], x=x[::-1])

print(f"Forward:  {forward}")
print(f"Backward: {backward}")
print(f"Expected: {-forward}")
print(f"Difference: {forward + backward}")
```

Output:
```
Forward:  0.4166666666666667
Backward: -0.3333333333333333
Expected: -0.4166666666666667
Difference: 0.08333333333333334
```

## Why This Is A Bug

The reversal property is a fundamental property of definite integrals: ∫ᵇₐ f(x)dx = -∫ₐᵇ f(x)dx. This property must hold for any numerical integration method to be mathematically correct. The `trapezoid` function in scipy.integrate correctly satisfies this property (as documented in its docstring), but `simpson` does not.

The bug occurs because when the number of sample points N is even, `simpson` uses an asymmetric correction formula (from Cartwright 2017) that is applied only to the last interval. When the data is reversed, this correction is applied to a different part of the integration range, leading to inconsistent results.

## Fix

The root cause is in the even-N case (lines 463-534 of `_quadrature.py`). The asymmetric correction formula violates the reversal property. The fix requires ensuring that the numerical integration method respects the fundamental property ∫ᵇₐ f(x)dx = -∫ₐᵇ f(x)dx.

One approach would be to apply symmetric corrections at both ends of the integration range when N is even, or use an alternative formulation that maintains the reversal property. The specific correction formula from Cartwright (lines 505-532) needs to be modified to be symmetric or replaced with an alternative approach that preserves this fundamental property.