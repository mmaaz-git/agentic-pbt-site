# Bug Report: pandas.core.window Rolling Variance Incorrect

**Target**: `pandas.core.window.Rolling.var()`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `rolling().var()` method produces incorrect variance values that can be 2x the correct value. The error appears to be caused by state from previous window calculations affecting subsequent windows, particularly when transitioning from large values to small values.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages')

import numpy as np
import pandas as pd
from hypothesis import given, strategies as st, assume, settings
from hypothesis.extra.pandas import series, range_indexes

@given(
    s=series(
        dtype=float,
        index=range_indexes(min_size=5, max_size=30),
        elements=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6)
    ),
    window=st.integers(min_value=2, max_value=10)
)
@settings(max_examples=500, deadline=None)
def test_rolling_variance_non_negative(s, window):
    assume(len(s) >= window)

    var = s.rolling(window=window).var()

    for i in range(len(s)):
        if not np.isnan(var.iloc[i]):
            assert var.iloc[i] >= -1e-10, \
                f"Variance is negative at index {i}: {var.iloc[i]}"
```

**Failing input**:
```python
s = pd.Series([0.0, 361328.203125, -862116.111476, 0.0, 0.015625])
window = 2
```

## Reproducing the Bug

```python
import pandas as pd

s = pd.Series([0.0, 361328.203125, -862116.111476, 0.0, 0.015625])

rolling_var = s.rolling(window=2).var()
window_at_4 = s.iloc[3:5]
manual_var_at_4 = window_at_4.var()

print(f"Window values at index 4: {list(window_at_4)}")
print(f"Manual variance: {manual_var_at_4}")
print(f"Rolling variance: {rolling_var.iloc[4]}")
print(f"Ratio: {rolling_var.iloc[4] / manual_var_at_4}")
```

Output:
```
Window values at index 4: [0.0, 0.015625]
Manual variance: 0.0001220703125
Rolling variance: 0.000244140625
Ratio: 2.0
```

## Why This Is A Bug

The rolling variance calculation produces a value that is exactly 2x the correct variance. Variance is a well-defined statistical measure: `var(X) = E[(X - μ)²]`, where μ is the mean.

For the window `[0.0, 0.015625]`:
- Mean = 0.0078125
- Variance = ((0 - 0.0078125)² + (0.015625 - 0.0078125)²) / 1 = 0.0001220703125

However, `rolling().var()` returns 0.000244140625, which is exactly double.

Interestingly, when this window is computed in isolation (not as part of a larger series), the variance is calculated correctly:

```python
pd.Series([0.0, 0.015625]).rolling(window=2).var().iloc[1]
# Returns: 0.0001220703125 (correct!)
```

This indicates that the bug is caused by state from previous window calculations affecting subsequent windows. The issue appears when transitioning from windows with large values (hundreds of thousands) to windows with small values (near zero).

## Impact

- **Data analysts** computing rolling statistics will get incorrect variance values
- **Financial applications** using rolling volatility calculations will produce wrong risk metrics
- **Scientific computing** relying on rolling variance for analysis will produce invalid results
- The bug is **silent** - no error is raised, making it difficult to detect
- The factor of 2x error is systematic and can compound in downstream calculations

## Fix

The bug appears to be in the rolling window variance calculation where state from previous windows is incorrectly carried over to subsequent calculations. The implementation likely maintains running statistics (sum, sum of squares, etc.) and fails to properly reset or update these values when the window slides.

The fix should ensure that each window's variance is calculated independently or that the running state is correctly maintained. Possible approaches:

1. **Independent calculation**: Calculate variance for each window from scratch without relying on state from previous windows
2. **Correct incremental update**: If using Welford's online algorithm or similar, ensure the state update correctly handles window boundaries
3. **Validation**: Add internal consistency checks to verify `var == std²` and that variance is always non-negative

The rolling variance implementation should be reviewed to identify where the 2x factor is introduced, likely in the handling of the sliding window transition.