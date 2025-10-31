# Bug Report: scipy.integrate.cumulative_simpson - Violates Monotonicity for Positive Functions

**Target**: `scipy.integrate.cumulative_simpson`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`cumulative_simpson` produces negative intermediate values when integrating a non-negative function, violating the fundamental property that the cumulative integral of a non-negative function must be monotonically non-decreasing.

## Property-Based Test

```python
import numpy as np
from hypothesis import assume, given, settings, strategies as st
from scipy import integrate


@settings(max_examples=300)
@given(
    st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6), min_size=3, max_size=50)
)
def test_cumulative_simpson_monotonic_for_positive_function(y_vals):
    y = np.abs(np.array(y_vals))
    assume(np.all(y >= 0))
    assume(len(y) >= 3)

    result = integrate.cumulative_simpson(y, initial=0)
    diffs = np.diff(result)

    assert np.all(diffs >= -1e-9)
```

**Failing input**: `y_vals=[0.0, 0.0, 1.0]`

## Reproducing the Bug

```python
import numpy as np
from scipy import integrate

y = np.array([0.0, 0.0, 1.0])
result = integrate.cumulative_simpson(y, initial=0)

print(f"y: {y}")
print(f"cumulative_simpson result: {result}")
print(f"Differences: {np.diff(result)}")
```

Output:
```
y: [0. 0. 1.]
cumulative_simpson result: [ 0.         -0.08333333  0.33333333]
Differences: [-0.08333333  0.41666667]
```

The cumulative integral goes negative (-0.0833) even though all y values are non-negative!

Comparison with `cumulative_trapezoid`:
```python
trap_result = integrate.cumulative_trapezoid(y, initial=0)
print(f"cumulative_trapezoid result: {trap_result}")
print(f"Differences: {np.diff(trap_result)}")
```

Output:
```
cumulative_trapezoid result: [0.  0.  0.5]
Differences: [0.  0.5]
```

`cumulative_trapezoid` correctly maintains non-negativity.

## Why This Is A Bug

For any non-negative function f(x) ≥ 0, the cumulative integral F(x) = ∫_a^x f(t)dt must be monotonically non-decreasing. This is a fundamental theorem of calculus. The cumulative Simpson's rule should preserve this property.

When the cumulative integral becomes negative for a positive function, it indicates a serious error in the implementation that could lead to:
- Silent data corruption in scientific computations
- Incorrect results in applications relying on monotonicity (e.g., cumulative distribution functions)
- Violation of physical constraints in simulations

## Fix

The issue appears to be in how `cumulative_simpson` handles the integration over subintervals. The Simpson's 1/3 rule formula being applied may not properly account for the contributions at each step, leading to incorrect intermediate values.

A proper implementation should ensure that:
1. Each step adds a non-negative contribution for non-negative functions
2. The formula gracefully handles edge cases (like constant or piecewise constant functions)
3. The implementation degrades to `cumulative_trapezoid` when Simpson's rule cannot be applied reliably (e.g., with fewer than 3 points or certain data patterns)