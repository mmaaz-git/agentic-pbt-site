# Bug Report: pandas Rolling Variance Returns Negative Values

**Target**: `pandas.core.window.rolling.Rolling.var`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The rolling variance calculation in pandas can return negative values, violating the mathematical property that variance must always be non-negative. This occurs when processing data containing very large values followed by much smaller values, likely due to numerical precision issues in the variance algorithm.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import pandas as pd

@given(
    st.lists(
        st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10),
        min_size=2,
        max_size=20
    ),
    st.integers(min_value=2, max_value=10)
)
@settings(max_examples=1000)
def test_rolling_variance_always_nonnegative(data, window):
    if window > len(data):
        return
    s = pd.Series(data)
    var = s.rolling(window=window).var()
    valid = var[~var.isna()]
    assert (valid >= 0).all(), f"Found negative variance: {valid[valid < 0]}"
```

**Failing input**: `data=[3222872787.0, 0.0, 2.0, 0.0], window=3`

## Reproducing the Bug

```python
import pandas as pd
import numpy as np

data = [3222872787.0, 0.0, 2.0, 0.0]
s = pd.Series(data)
var = s.rolling(window=3).var()

print(var)
```

**Output:**
```
0         NaN
1         NaN
2    5.201208e+18
3   -5.116667e+02
dtype: float64
```

The variance at index 3 is **-511.67**, which is impossible since variance must be ≥ 0.

**Manual verification for window [0.0, 2.0, 0.0]:**
- Mean = (0 + 2 + 0) / 3 = 0.667
- Variance = [(0-0.667)² + (2-0.667)² + (0-0.667)²] / 2 = 1.333

NumPy correctly computes: `np.var([0.0, 2.0, 0.0], ddof=1) = 1.333`

## Why This Is A Bug

Variance is mathematically defined as the average squared deviation from the mean, which **must always be non-negative**. Negative variance violates this fundamental property and produces silently incorrect results that can corrupt downstream statistical analyses.

The bug appears when:
1. Large magnitude values (e.g., 3.2 billion) are followed by
2. Much smaller values (e.g., 0, 1, 2)

This suggests a numerical precision issue, likely catastrophic cancellation in a two-pass or online variance algorithm when the running sum of squares loses precision due to the large initial values.

## Fix

The root cause is in the Cython implementation of rolling variance (`pandas._libs.window.aggregations`). The fix requires using a numerically stable variance algorithm such as:

1. **Welford's online algorithm** - accumulates variance incrementally without catastrophic cancellation
2. **Compensated summation** (Kahan summation) - reduces accumulated floating-point errors

The current implementation likely uses a formula like `var = E[X²] - E[X]²`, which is known to be numerically unstable when E[X²] and E[X]² are close in magnitude.

A stable two-pass algorithm:
```python
mean = sum(x) / n
variance = sum((x - mean)**2) / (n - 1)
```

Or Welford's one-pass algorithm:
```python
M = 0
S = 0
for i, x in enumerate(values, 1):
    old_M = M
    M += (x - M) / i
    S += (x - M) * (x - old_M)
variance = S / (n - 1)
```

Since the implementation is in Cython (`_libs/window/aggregations.pyx`), the fix requires modifying the low-level aggregation code to use a numerically stable algorithm.