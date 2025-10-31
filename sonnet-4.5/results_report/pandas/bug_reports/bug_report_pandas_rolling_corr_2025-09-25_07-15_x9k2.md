# Bug Report: pandas.core.window Rolling.corr Infinite Correlation

**Target**: `pandas.core.window.rolling.Rolling.corr`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `Rolling.corr()` method returns infinite correlation values when computing correlation on data with zero or near-zero variance, instead of returning NaN.

## Property-Based Test

```python
import numpy as np
import pandas as pd
from hypothesis import given, strategies as st, assume, settings


@settings(max_examples=200)
@given(
    st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e3, max_value=1e3), min_size=5, max_size=50),
    st.integers(min_value=2, max_value=10)
)
def test_rolling_corr_bounds(data, window):
    assume(len(data) >= window)
    assume(len(set(data)) > 1)

    s = pd.Series(data)
    corr = s.rolling(window=window).corr()

    valid = corr[~corr.isna()]
    if len(valid) > 0:
        assert (valid >= -1.0001).all(), f"Found correlation < -1: {valid.min()}"
        assert (valid <= 1.0001).all(), f"Found correlation > 1: {valid.max()}"
```

**Failing input**: `data=[0.0, 0.0, 0.0, 0.0, 7.797011399495068e-124], window=2`

## Reproducing the Bug

```python
import pandas as pd

data = [0.0, 0.0, 0.0, 0.0, 7.797e-124]
s = pd.Series(data)

corr = s.rolling(window=2).corr()
print(corr)
```

Output:
```
0    NaN
1    NaN
2    NaN
3    NaN
4    inf
dtype: float64
```

The correlation at index 4 is `inf`, but mathematically correlation must be in the range [-1, 1]. When the variance is zero or near-zero (constant or nearly constant values), the correlation is undefined and should be NaN.

## Why This Is A Bug

Correlation is defined as:
```
corr(X, Y) = cov(X, Y) / (std(X) * std(Y))
```

When `std(X)` or `std(Y)` is zero (which happens when all values in the window are identical or nearly identical), the denominator becomes zero, making the correlation undefined. The mathematically correct result is NaN, not infinity.

In this case, the window `[0.0, 0.0]` has zero variance, so the correlation should be NaN. The division by zero produces infinity instead.

## Fix

```diff
--- a/pandas/core/window/rolling.py
+++ b/pandas/core/window/rolling.py
@@ -1843,7 +1843,10 @@ class RollingAndExpandingMixin(BaseWindow):
                 numerator = (mean_x_y - mean_x * mean_y) * (
                     count_x_y / (count_x_y - ddof)
                 )
                 denominator = (x_var * y_var) ** 0.5
-                result = numerator / denominator
+                # Set result to NaN where denominator is zero or near-zero
+                result = np.where(
+                    np.isclose(denominator, 0.0, atol=1e-14), np.nan, numerator / denominator
+                )
             return Series(result, index=x.index, name=x.name, copy=False)
```

Alternatively, the code could check if variance is zero before computing correlation and return NaN in those cases.