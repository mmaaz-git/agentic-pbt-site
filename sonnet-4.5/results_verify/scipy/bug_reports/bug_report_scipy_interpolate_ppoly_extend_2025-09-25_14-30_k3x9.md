# Bug Report: scipy.interpolate.PPoly.extend Modifies Existing Values

**Target**: `scipy.interpolate.PPoly.extend`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `PPoly.extend` method incorrectly modifies the polynomial's value at the boundary point between the original and extended domains, violating the expected behavior that extending should only add new intervals without changing existing values.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import numpy as np
import scipy.interpolate


@given(st.integers(min_value=3, max_value=10))
@settings(max_examples=300)
def test_ppoly_extend_consistency(num_intervals):
    """
    Property: PPoly.extend should preserve existing polynomial values.

    Evidence: extend() method documentation says it should "Add additional
    breakpoints and coefficients" without mentioning it modifies existing values.
    """
    degree = 2
    x1 = np.linspace(0, 1, num_intervals + 1)
    c1 = np.random.randn(degree + 1, num_intervals)

    ppoly1 = scipy.interpolate.PPoly(c1, x1)
    test_points = np.linspace(0, 1, 10)
    vals_before = ppoly1(test_points)

    x_ext = np.array([1.5])
    c_ext = np.random.randn(degree + 1, 1)

    ppoly1.extend(c_ext, x_ext)
    vals_after = ppoly1(test_points)

    assert np.allclose(vals_before, vals_after, rtol=1e-10, atol=1e-10), \
        f"PPoly.extend should not change values in original domain"
```

**Failing input**: `num_intervals=3`

## Reproducing the Bug

```python
import numpy as np
import scipy.interpolate

np.random.seed(0)

num_intervals = 3
degree = 2
x1 = np.linspace(0, 1, num_intervals + 1)
c1 = np.random.randn(degree + 1, num_intervals)

ppoly1 = scipy.interpolate.PPoly(c1, x1)

print(f"Before extend: f(1.0) = {ppoly1(1.0)}")

x_ext = np.array([1.5])
c_ext = np.random.randn(degree + 1, 1)

ppoly1.extend(c_ext, x_ext)

print(f"After extend:  f(1.0) = {ppoly1(1.0)}")
print(f"Value changed by: {ppoly1(1.0) - (-0.32022948)}")
```

Output:
```
Before extend: f(1.0) = -0.32022948
After extend:  f(1.0) = 1.45427351
Value changed by: 1.77450299
```

## Why This Is A Bug

When `PPoly.extend` adds a new interval starting at `x=1.0`, the evaluation at that boundary point switches from using the last coefficient of the original polynomial (interval [0.667, 1.0]) to using the first coefficient of the extended polynomial (interval [1.0, 1.5]). This violates the fundamental expectation that:

1. The `extend` method should not modify values in the original domain `[0, 1]`
2. Piecewise polynomials should maintain continuity (or at least consistent evaluation) at interior breakpoints

The issue appears to be in how `PPoly.__call__` determines which interval to use when evaluating at a boundary point. When `x=1.0` becomes the start of a new interval after extension, the evaluation incorrectly uses the new interval's constant term instead of the original interval's value at that point.

## Fix

The bug is in the interval selection logic during evaluation. At boundary points that are both the end of one interval and the start of another, PPoly should consistently use the left interval for evaluation to maintain continuity. The fix would be in the `PPoly.__call__` or the underlying evaluation method to ensure that when `x[i] == evaluation_point`, it uses interval `i-1` (if it exists) rather than interval `i`.

A simpler fix might be to document this as expected behavior and require users to ensure continuity manually, but this would be surprising and error-prone. The mathematically correct fix is to use left-continuity at interior breakpoints.