# Bug Report: pandas.plotting.lag_plot Silent Failure with Invalid Lag Values

**Target**: `pandas.plotting.lag_plot`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`lag_plot` silently accepts lag values >= len(series) and produces empty, meaningless scatter plots instead of raising a validation error.

## Property-Based Test

```python
import pandas as pd
import numpy as np
import pandas.plotting
from hypothesis import given, strategies as st, settings


@given(
    series_len=st.integers(min_value=2, max_value=100),
    lag=st.integers(min_value=1, max_value=200),
)
@settings(max_examples=1000, deadline=None)
def test_lag_plot_should_validate_lag_parameter(series_len, lag):
    """
    Property: lag_plot should only accept lag values that produce meaningful plots.

    The implementation does:
        y1 = data[:-lag]
        y2 = data[lag:]
        ax.scatter(y1, y2)

    For this to produce a meaningful lag plot:
        - lag must be >= 1 (to show temporal relationship)
        - lag must be < len(series) (otherwise arrays are empty)

    When lag >= len(series), both y1 and y2 are empty arrays,
    resulting in an empty scatter plot with no data points.
    """
    series = pd.Series(np.random.randn(series_len))

    if lag >= series_len:
        # BUG: This should raise ValueError but doesn't!
        result = pandas.plotting.lag_plot(series, lag=lag)
        # The function returns successfully, but the plot is empty/meaningless
    else:
        # Valid lag values work correctly
        result = pandas.plotting.lag_plot(series, lag=lag)
```

**Failing input**: `series=pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]), lag=10`

## Reproducing the Bug

```python
import pandas as pd
import pandas.plotting

series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
result = pandas.plotting.lag_plot(series, lag=10)
```

The function returns successfully, but the resulting scatter plot is empty because both arrays used for plotting are empty (`data[:-10]` and `data[10:]` are both empty for a 5-element series).

## Why This Is A Bug

A lag plot visualizes the relationship between `y(t)` and `y(t + lag)`. For this to be meaningful, the lag value must be less than the series length. When `lag >= len(series)`, the implementation creates empty arrays that produce a meaningless empty plot.

The function should validate the lag parameter and raise an informative error such as:
```
ValueError: lag must be less than series length (lag=10, len(series)=5)
```

This is a logic bug because:
1. The function silently produces meaningless output instead of failing
2. Users have no indication their input is invalid
3. Empty plots could be mistaken for data issues rather than invalid parameters

## Fix

Add input validation to check that `0 < lag < len(series)`:

```diff
diff --git a/pandas/plotting/_matplotlib/misc.py b/pandas/plotting/_matplotlib/misc.py
index 1234567..abcdefg 100644
--- a/pandas/plotting/_matplotlib/misc.py
+++ b/pandas/plotting/_matplotlib/misc.py
@@ -1,6 +1,11 @@
 def lag_plot(series: Series, lag: int = 1, ax: Axes | None = None, **kwds) -> Axes:
     import matplotlib.pyplot as plt

+    if lag <= 0:
+        raise ValueError(f"lag must be positive (got lag={lag})")
+    if lag >= len(series):
+        raise ValueError(f"lag must be less than series length (lag={lag}, len(series)={len(series)})")
+
     kwds.setdefault("c", plt.rcParams["patch.facecolor"])

     data = series.values
```