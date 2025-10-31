# Bug Report: scipy.integrate.simpson - Violates Reversal Property

**Target**: `scipy.integrate.simpson`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`scipy.integrate.simpson` violates the fundamental reversal property of integration: ∫_a^b f(x)dx = -∫_b^a f(x)dx. Integrating in the forward direction produces a different magnitude than integrating in the reverse direction.

## Property-Based Test

```python
import math
import numpy as np
from hypothesis import assume, given, settings, strategies as st
from scipy import integrate


@settings(max_examples=300)
@given(
    st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=0.01, max_value=1e6), min_size=3, max_size=50),
    st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6), min_size=3, max_size=50)
)
def test_simpson_reversal(x_vals, y_vals):
    assume(len(x_vals) == len(y_vals))
    x = np.sort(np.array(x_vals))
    y = np.array(y_vals)
    assume(len(np.unique(x)) == len(x))

    result_forward = integrate.simpson(y, x=x)
    result_backward = integrate.simpson(y[::-1], x=x[::-1])

    assert math.isclose(result_forward, -result_backward, rel_tol=1e-9, abs_tol=1e-9)
```

**Failing input**: `x_vals=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], y_vals=[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]`

## Reproducing the Bug

```python
import numpy as np
from scipy import integrate

x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
y = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

result_forward = integrate.simpson(y, x=x)
result_backward = integrate.simpson(y[::-1], x=x[::-1])

print(f"Forward integration: {result_forward}")
print(f"Backward integration: {result_backward}")
print(f"Negated backward: {-result_backward}")
print(f"Match? {np.isclose(result_forward, -result_backward)}")
```

Output:
```
Forward integration: 0.4166666666666667
Backward integration: -0.3333333333333333
Negated backward: 0.3333333333333333
Match? False
```

The forward integral (0.4167) does not equal the negated backward integral (0.3333).

Comparison with `trapezoid`:
```python
trap_forward = integrate.trapezoid(y, x=x)
trap_backward = integrate.trapezoid(y[::-1], x=x[::-1])

print(f"Trapezoid forward: {trap_forward}")
print(f"Trapezoid backward: {trap_backward}")
print(f"Match? {np.isclose(trap_forward, -trap_backward)}")
```

Output:
```
Trapezoid forward: 0.5
Trapezoid backward: -0.5
Match? True
```

`trapezoid` correctly satisfies the reversal property.

## Why This Is A Bug

The reversal property ∫_a^b f(x)dx = -∫_b^a f(x)dx is a fundamental property of integration. Any numerical integration method should preserve this property (up to numerical precision).

Simpson's rule implementation appears to be asymmetric in how it handles the endpoints or subintervals, causing different results when the data is reversed. This can lead to:
- Inconsistent results depending on the order of data
- Errors in applications that rely on the reversal property
- Loss of trust in numerical integration results

## Fix

The issue likely stems from how Simpson's rule handles odd/even numbers of intervals or the special treatment of endpoints. When integrating backward, the algorithm may apply different corrections or formulas at the boundaries.

The implementation should ensure that:
1. The algorithm is symmetric with respect to direction
2. Endpoint corrections are applied consistently regardless of integration direction
3. The composite Simpson's rule formula is applied uniformly

The specific fix requires examining the implementation in `/scipy/integrate/_quadrature.py` to identify where the asymmetry occurs, particularly in the handling of the last intervals and the Cartwright correction formula.