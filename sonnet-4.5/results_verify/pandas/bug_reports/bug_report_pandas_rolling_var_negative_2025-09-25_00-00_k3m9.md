# Bug Report: pandas.core.window.Rolling.var() Returns Negative Variance

**Target**: `pandas.core.window.Rolling.var()`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `rolling().var()` method can return negative variance values due to numerical stability issues, violating the fundamental mathematical property that variance must be non-negative.

## Property-Based Test

```python
import numpy as np
import pandas as pd
from hypothesis import given, strategies as st, settings, assume


@given(
    series=st.lists(
        st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
        min_size=5, max_size=50
    ),
    window=st.integers(min_value=2, max_value=10)
)
@settings(max_examples=200)
def test_rolling_var_non_negative(series, window):
    assume(len(series) >= window)

    s = pd.Series(series)
    rolling_var = s.rolling(window=window).var()

    valid_mask = ~rolling_var.isna()
    if valid_mask.sum() > 0:
        assert (rolling_var[valid_mask] >= -1e-10).all(), \
            f"Variance should be non-negative, got {rolling_var[valid_mask].min()}"
```

**Failing input**: `series=[0.0, 494699.5, 0.0, 0.0, 0.00390625], window=3`

## Reproducing the Bug

```python
import pandas as pd
import numpy as np

series = [0.0, 494699.5, 0.0, 0.0, 0.00390625]
s = pd.Series(series)

rolling_var = s.rolling(window=3).var()
negative_var = rolling_var.iloc[4]

print(f"Rolling variance at index 4: {negative_var}")
print(f"Window data: {series[2:5]}")
print(f"Expected (numpy): {np.var(series[2:5], ddof=1)}")
print(f"BUG: Variance is negative: {negative_var < 0}")
```

Output:
```
Rolling variance at index 4: -1.017252596587544e-05
Window data: [0.0, 0.0, 0.00390625]
Expected (numpy): 5.086263020833334e-06
BUG: Variance is negative: True
```

## Why This Is A Bug

Variance is mathematically defined as the average of squared differences from the mean, which means it must always be non-negative (≥ 0). This bug violates that fundamental property.

The issue appears when computing variance over windows with values that vary greatly in magnitude, causing numerical instability in the variance calculation. The correct variance for `[0.0, 0.0, 0.00390625]` with `ddof=1` is approximately `5.086e-06`, but pandas computes `-1.017e-05`.

This is a high-severity logic bug because:
- It violates a fundamental mathematical invariant
- It can produce incorrect statistical analysis results
- The error can propagate to dependent calculations (e.g., `std = sqrt(var)` would fail)
- Users relying on rolling variance for data analysis will get silently corrupted results

## Fix

This is likely a numerical stability issue in the Cython implementation of rolling variance. The fix would require examining the variance calculation algorithm in `pandas/_libs/window/aggregations.pyx` and using a numerically stable algorithm such as Welford's online algorithm or the two-pass algorithm with compensated summation.

A potential approach:
1. Identify the variance calculation in the Cython backend
2. Replace with a numerically stable variance algorithm
3. Add explicit checks to ensure variance is never negative (clamp to 0 if needed)
4. Add regression tests for this specific case