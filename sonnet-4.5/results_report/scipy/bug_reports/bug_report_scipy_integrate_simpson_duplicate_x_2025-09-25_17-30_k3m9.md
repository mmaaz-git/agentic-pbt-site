# Bug Report: scipy.integrate.simpson - Incorrect Results with Duplicate X Values

**Target**: `scipy.integrate.simpson`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`scipy.integrate.simpson` produces incorrect integration results when the x array contains duplicate values (zero-width intervals). For a constant function, the integral should equal c*(b-a), but simpson returns incorrect values while trapezoid correctly handles this case.

## Property-Based Test

```python
import math
import numpy as np
from hypothesis import assume, given, settings, strategies as st
from scipy import integrate


@settings(max_examples=500)
@given(
    st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=0.01, max_value=1e6), min_size=3, max_size=100),
    st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6)
)
def test_simpson_constant(x_vals, c):
    x = np.sort(np.array(x_vals))
    y = np.full_like(x, c)

    result = integrate.simpson(y, x=x)
    expected = c * (x[-1] - x[0])

    assert math.isclose(result, expected, rel_tol=1e-9, abs_tol=1e-9)
```

**Failing input**: `x_vals=[1.0, 1.0, 2.0], c=1.0`

## Reproducing the Bug

```python
import numpy as np
from scipy import integrate

x = np.array([1.0, 1.0, 2.0])
y = np.array([1.0, 1.0, 1.0])

result = integrate.simpson(y, x=x)
expected = 1.0

print(f"simpson result: {result}")
print(f"Expected: {expected}")
print(f"trapezoid result: {integrate.trapezoid(y, x=x)}")
```

Output:
```
simpson result: 0.6666666666666666
Expected: 1.0
trapezoid result: 1.0
```

Additional failing cases:
- `x=[0.0, 0.0, 1.0], y=[2.0, 2.0, 2.0]` → simpson returns 1.333 instead of 2.0
- `x=[0.0, 0.0, 0.0, 1.0], y=[5.0, 5.0, 5.0, 5.0]` → simpson returns 1.667 instead of 5.0
- `x=[1.0, 2.0, 2.0], y=[3.0, 3.0, 3.0]` → simpson returns 2.0 instead of 3.0

## Why This Is A Bug

The integral of a constant function y=c over interval [a,b] should always equal c*(b-a), regardless of how the interval is sampled. This is a fundamental mathematical property. When x contains duplicate values (representing zero-width intervals), simpson should either:
1. Handle them correctly (like trapezoid does), or
2. Raise an error indicating invalid input

Instead, it silently returns incorrect results. The trapezoid method correctly handles duplicate x values, proving this is solvable.

## Fix

The issue likely lies in how simpson handles the spacing between points when computing the composite formula. When consecutive x values are identical (dx=0), the current implementation doesn't properly account for these zero-width intervals.

A high-level fix would be to:
1. Detect when consecutive x values are identical
2. Either skip those intervals (treat them as having zero contribution) or collapse them into a single point
3. Ensure the algorithm degrades gracefully to trapezoid when appropriate

The specific fix requires examining the implementation in `/scipy/integrate/_quadrature.py`, particularly the `_basic_simpson` function and how it handles non-uniform spacing.