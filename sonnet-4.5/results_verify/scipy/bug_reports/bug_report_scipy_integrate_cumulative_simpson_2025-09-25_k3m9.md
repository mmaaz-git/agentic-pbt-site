# Bug Report: scipy.integrate.cumulative_simpson - Incorrect Intermediate Values

**Target**: `scipy.integrate.cumulative_simpson`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `cumulative_simpson` function produces mathematically incorrect intermediate cumulative integral values. For non-negative functions, it can produce negative cumulative values, and for simple linear functions, intermediate values are wrong by a factor of 2.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/scipy_env')

import numpy as np
import scipy.integrate as integrate
from hypothesis import given, strategies as st, settings
import math

@given(
    st.lists(st.floats(min_value=0, max_value=1e6, allow_nan=False, allow_infinity=False),
             min_size=3, max_size=100)
)
@settings(max_examples=1000)
def test_cumulative_simpson_monotonic_for_positive(y_list):
    y = np.array(y_list)
    cumulative = integrate.cumulative_simpson(y, initial=0)

    for i in range(len(cumulative) - 1):
        assert cumulative[i] <= cumulative[i+1], \
            f"Monotonicity violated at index {i}: {cumulative[i]} > {cumulative[i+1]}"
```

**Failing input**: `[0.0, 0.0, 1.0]`

## Reproducing the Bug

```python
import numpy as np
from scipy.integrate import cumulative_simpson

y = np.array([0.0, 0.0, 1.0])
result = cumulative_simpson(y, initial=0)
print(f"Input: {y}")
print(f"Result: {result}")

y2 = np.array([0.0, 0.5, 1.0])
result2 = cumulative_simpson(y2, initial=0)
print(f"\nInput (linear y=x): {y2}")
print(f"Result: {result2}")
print(f"Expected: [0.0, 0.125, 0.5]")
```

Output:
```
Input: [0.0 0.0 1.0]
Result: [ 0.         -0.08333333  0.33333333]

Input (linear y=x): [0.  0.5 1. ]
Result: [0.   0.25 1.  ]
Expected: [0.0, 0.125, 0.5]
```

## Why This Is A Bug

1. **Negative values for non-negative functions**: The cumulative integral of a non-negative function must be non-negative. For input `[0.0, 0.0, 1.0]`, the function produces `result[1] = -0.0833...`, which is mathematically impossible.

2. **Incorrect values for linear functions**: For the linear function y=x sampled at `[0, 0.5, 1]`, the integral from 0 to 0.5 should be `∫₀^0.5 x dx = 0.5²/2 = 0.125`, but `cumulative_simpson` returns `0.25` (exactly double the correct value).

3. **Non-monotonic cumulative integrals**: For input `[1.0, 0.0, 0.0]`, the cumulative values are `[0.0, 0.4166..., 0.3333...]`, where the cumulative decreases from index 1 to 2, violating the fundamental property that cumulative integrals of non-negative functions must be monotonically non-decreasing.

The final cumulative value (total integral) appears to be correct, but all intermediate values are wrong. This suggests that the algorithm is using future points to estimate past cumulative values, which violates the definition of cumulative integration.

## Fix

The issue appears to be in how `cumulative_simpson` computes intermediate values. The function should compute the cumulative integral up to each point using only the data up to that point, but it seems to be using a global fit that incorporates future points.

For cumulative integration with Simpson's rule, at points where a full Simpson's triplet isn't available (e.g., at the second point when only two points are available), the algorithm should fall back to a method that only uses available past data, such as the trapezoidal rule.

A proper fix would ensure that:
1. `cumulative[i]` only depends on `y[0:i+1]`
2. For points where Simpson's rule cannot be properly applied (first few points), use trapezoidal rule
3. The final value should still match the result of `simpson(y)`

Without access to modify the source, a high-level fix strategy would be:
- Review the current algorithm's use of future points in computing cumulative values
- Implement a causal algorithm that computes each cumulative value using only past data
- Ensure compatibility with the existing `simpson()` function's total integral