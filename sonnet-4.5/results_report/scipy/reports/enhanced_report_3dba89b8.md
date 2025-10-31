# Bug Report: scipy.integrate.cumulative_simpson Monotonicity Violation

**Target**: `scipy.integrate.cumulative_simpson`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `cumulative_simpson` function violates the fundamental mathematical property that the cumulative integral of a non-negative function must be monotonically non-decreasing, producing negative differences in the cumulative integral even when integrating strictly non-negative functions.

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

<details>

<summary>
**Failing input**: `y=[0.0, 0.0, 1.0]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/59/hypo.py", line 21, in <module>
    test_cumulative_simpson_monotonic_increasing_for_positive()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/59/hypo.py", line 6, in test_cumulative_simpson_monotonic_increasing_for_positive
    st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/59/hypo.py", line 18, in test_cumulative_simpson_monotonic_increasing_for_positive
    assert np.all(diffs >= -1e-10)
           ~~~~~~^^^^^^^^^^^^^^^^^
AssertionError
Falsifying example: test_cumulative_simpson_monotonic_increasing_for_positive(
    y=[0.0, 0.0, 1.0],
)
```
</details>

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

<details>

<summary>
Cumulative integral incorrectly goes negative
</summary>
```
y = [0. 0. 1.]
cumulative_simpson(y, initial=0) = [ 0.         -0.08333333  0.33333333]
Differences between consecutive values: [-0.08333333  0.41666667]
Has negative difference: True
```
</details>

## Why This Is A Bug

This violates the fundamental theorem of calculus which guarantees that for any non-negative function f(x) ≥ 0, the cumulative integral F(x) = ∫₀ˣ f(t)dt must be monotonically non-decreasing. In mathematical terms, F'(x) = f(x) ≥ 0 implies F(x₂) ≥ F(x₁) for all x₂ > x₁.

In the failing example, we're integrating a non-negative function [0, 0, 1] but the cumulative result contains a negative value (-0.08333333), which is mathematically impossible. The cumulative integral should never decrease when integrating non-negative values. This is not a numerical precision issue but a fundamental algorithmic error in how Simpson's rule is being applied cumulatively.

## Relevant Context

The bug originates from the asymmetric correction formulas used in `_cumulative_simpson_equal_intervals` (line 567-578 in `/home/npc/pbt/agentic-pbt/envs/scipy_env/lib/python3.13/site-packages/scipy/integrate/_quadrature.py`). The formula from equation (10) of Cartwright's paper includes the term `5*f1/4 + 2*f2 - f3/4`, where the negative coefficient for f3 can produce negative sub-integrals even when all function values are non-negative.

The implementation alternates between h1 and h2 intervals (forward and reverse) to compute cumulative integrals, but these asymmetric formulas don't preserve monotonicity. The same issue likely affects `_cumulative_simpson_unequal_intervals` which uses equation (8) from the same paper.

Documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.cumulative_simpson.html
Source code: scipy/integrate/_quadrature.py:612-793

## Proposed Fix

The fix requires ensuring that cumulative integrals of non-negative functions remain non-decreasing. One approach is to use modified formulas that guarantee monotonicity or add post-processing to enforce this mathematical constraint. Here's a high-level approach:

1. Detect when the integrand is non-negative
2. Apply a monotonicity-preserving correction when negative differences occur
3. Consider falling back to cumulative_trapezoid for edge cases where Simpson's rule produces invalid results
4. Alternatively, use symmetric formulas that don't introduce negative coefficients

The current implementation's reliance on asymmetric formulas with negative coefficients fundamentally causes this issue. A proper fix would require either:
- Using different integration formulas that preserve monotonicity
- Adding validation and correction steps to ensure mathematical validity
- Documenting this as a known limitation and recommending cumulative_trapezoid when monotonicity is required