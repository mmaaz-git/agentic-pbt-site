# Bug Report: scipy.integrate.cumulative_simpson Monotonicity Violation

**Target**: `scipy.integrate.cumulative_simpson`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `cumulative_simpson` function violates the fundamental property that the cumulative integral of a non-negative function must be monotonically non-decreasing. This results in negative differences in the cumulative integral even when integrating a non-negative function.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume, settings
import numpy as np
from scipy.integrate import cumulative_simpson

@given(
    st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
             min_size=3, max_size=100)
)
@settings(max_examples=500)
def test_cumulative_simpson_monotonic_increasing_for_positive(y):
    y_arr = np.array(y)
    assume(np.all(y_arr >= 0))
    assume(np.any(y_arr > 0))

    cumulative_result = cumulative_simpson(y_arr, initial=0)

    diffs = np.diff(cumulative_result)
    assert np.all(diffs >= -1e-10)
```

**Failing input**: `y = [0.0, 0.0, 1.0]`

## Reproducing the Bug

```python
from scipy.integrate import cumulative_simpson
import numpy as np

y = np.array([0.0, 0.0, 1.0])

cumulative_result = cumulative_simpson(y, initial=0)
diffs = np.diff(cumulative_result)

print(f"y = {y}")
print(f"cumulative_simpson(y, initial=0) = {cumulative_result}")
print(f"Differences between consecutive values: {diffs}")
print(f"Has negative difference: {np.any(diffs < 0)}")
```

Output:
```
y = [0.0 0.0 1.0]
cumulative_simpson(y, initial=0) = [ 0.         -0.08333333  0.33333333]
Differences between consecutive values: [-0.08333333  0.41666667]
Has negative difference: True
```

## Why This Is A Bug

For any non-negative function f(x) ≥ 0, the cumulative integral F(x) = ∫₀ˣ f(t)dt must be monotonically non-decreasing, meaning F(x₂) ≥ F(x₁) for all x₂ > x₁. This is because we are accumulating non-negative contributions.

The bug manifests when `cumulative_simpson` produces a cumulative result where F(x₁) > F(x₂) for some x₂ > x₁, even though the integrand is non-negative everywhere. This violates a fundamental mathematical property of integration.

The `cumulative_trapezoid` function correctly respects this property, but `cumulative_simpson` does not.

## Fix

This bug is likely related to the asymmetric correction formulas used in the underlying `simpson` implementation (which has a separate reversal property bug). The `cumulative_simpson` function uses helper functions `_cumulative_simpson_equal_intervals` and `_cumulative_simpson_unequal_intervals` along with `_cumulatively_sum_simpson_integrals`.

The root cause appears to be in how Simpson's rule is applied cumulatively. The correction formulas used may not preserve monotonicity when applied in a cumulative fashion. A fix would require ensuring that cumulative sums of non-negative integrands remain non-decreasing, possibly by using symmetric correction formulas or alternative approaches that preserve this fundamental property.