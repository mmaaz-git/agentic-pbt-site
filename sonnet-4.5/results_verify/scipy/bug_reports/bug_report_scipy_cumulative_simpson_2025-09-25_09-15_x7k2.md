# Bug Report: scipy.integrate.cumulative_simpson Monotonicity Violation

**Target**: `scipy.integrate.cumulative_simpson`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`cumulative_simpson` violates the fundamental mathematical property that integrating a non-negative function must produce non-decreasing cumulative values. The function produces negative intermediate integrals when given certain non-negative inputs.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy as np
from scipy.integrate import cumulative_simpson


@given(
    st.lists(st.floats(min_value=0, max_value=1e6, allow_nan=False, allow_infinity=False),
             min_size=3, max_size=20)
)
def test_cumulative_simpson_monotonic(values):
    y = np.array(values)
    y_int = cumulative_simpson(y, initial=0)

    diffs = np.diff(y_int)
    assert np.all(diffs >= -1e-9), \
        f"Cumulative integral decreased for non-negative input y={values}"
```

**Failing input**: `values=[0.0, 0.0, 1.0]`

## Reproducing the Bug

```python
import numpy as np
from scipy.integrate import cumulative_simpson

y = np.array([0.0, 0.0, 1.0])
result = cumulative_simpson(y, initial=0)

print(f"Input (non-negative): {y}")
print(f"Cumulative integral: {result}")
print(f"Result at index 1: {result[1]}")

assert result[1] >= 0, \
    f"Integral of non-negative function cannot be negative! Got {result[1]}"
```

Output:
```
Input (non-negative): [0. 0. 1.]
Cumulative integral: [ 0.         -0.08333333  0.33333333]
Result at index 1: -0.08333333333333333
AssertionError: Integral of non-negative function cannot be negative! Got -0.08333333333333333
```

## Why This Is A Bug

When integrating a non-negative function y(x) ≥ 0, the cumulative integral must be monotonically non-decreasing. This is a fundamental property: ∫₀ˣ f(t)dt is increasing when f(t) ≥ 0.

However, `cumulative_simpson([0.0, 0.0, 1.0])` produces `[0.0, -0.083333, 0.333333]`, where the second value is **negative**. This violates the mathematical definition of integration.

For comparison, `cumulative_trapezoid` correctly produces `[0.0, 0.0, 0.5]` for the same input - all non-negative values.

The root cause is in `_cumulative_simpson_equal_intervals` at line 567-578 of `_quadrature.py`:

```python
def _cumulative_simpson_equal_intervals(y: np.ndarray, dx: np.ndarray) -> np.ndarray:
    d = dx[..., :-1]
    f1 = y[..., :-2]
    f2 = y[..., 1:-1]
    f3 = y[..., 2:]

    return d / 3 * (5 * f1 / 4 + 2 * f2 - f3 / 4)
```

The formula `(5*f1/4 + 2*f2 - f3/4)` has a negative coefficient for `f3`, which produces negative sub-interval integrals even when all values are non-negative. For y=[0,0,1], this gives: `1/3 * (0 + 0 - 1/4) = -1/12 ≈ -0.0833`.

## Fix

The issue stems from using an asymmetric approximation formula that doesn't preserve monotonicity. The implementation should use a formula that guarantees non-negative sub-interval integrals for non-negative functions.

One approach is to modify the algorithm to ensure monotonicity is preserved, similar to how `cumulative_trapezoid` works. Alternatively, the formula coefficients need adjustment to prevent negative contributions.

The trapezoidal rule formula at each sub-interval naturally preserves non-negativity:
```python
d * (f1 + f2) / 2
```

While the current Simpson's formula does not:
```python
d / 3 * (5 * f1 / 4 + 2 * f2 - f3 / 4)
```

A potential fix would be to use a different formulation that maintains the higher-order accuracy of Simpson's rule while preserving monotonicity, or to document this limitation clearly and fall back to `cumulative_trapezoid` when monotonicity is required.