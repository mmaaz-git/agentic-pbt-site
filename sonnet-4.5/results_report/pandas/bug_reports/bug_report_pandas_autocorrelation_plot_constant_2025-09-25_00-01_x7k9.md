# Bug Report: pandas.plotting.autocorrelation_plot produces NaN for constant series

**Target**: `pandas.plotting.autocorrelation_plot`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`autocorrelation_plot` silently produces a meaningless plot with all NaN values when given a constant series, without any warning or error. The root cause is division by zero when calculating the variance.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

@settings(max_examples=100, deadline=None)
@given(
    value=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
    length=st.integers(min_value=2, max_value=100)
)
def test_autocorrelation_plot_constant_series(value, length):
    series = pd.Series([value] * length)
    result = pd.plotting.autocorrelation_plot(series)

    lines = result.get_lines()
    if lines:
        ydata = lines[-1].get_ydata()
        assert not np.all(np.isnan(ydata)), "All autocorrelation values are NaN"

    plt.close('all')
```

**Failing input**: Any constant series

## Reproducing the Bug

```python
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')

series = pd.Series([42.0] * 20)
result = pd.plotting.autocorrelation_plot(series)

lines = result.get_lines()
autocorr_line = lines[-1]
ydata = autocorr_line.get_ydata()
print(f"Autocorrelation values: {ydata}")
print(f"All NaN: {np.all(np.isnan(ydata))}")
```

Output:
```
Autocorrelation values: [nan nan nan nan nan ...]
All NaN: True
```

## Why This Is A Bug

1. Constant series are valid inputs (e.g., from real-world data with no variation)
2. The function silently produces a meaningless plot without any warning or error
3. Expected behavior: either raise an informative error or return a meaningful result (autocorrelation undefined or 1.0)
4. The root cause is division by zero: `c0 = sum((data - mean)**2) / n = 0` for constant series

## Fix

Add validation to detect constant series and either raise an error or handle them specially:

```diff
--- a/pandas/plotting/_matplotlib/misc.py
+++ b/pandas/plotting/_matplotlib/misc.py
@@ -451,6 +451,11 @@ def autocorrelation_plot(series: Series, ax: Axes | None = None, **kwds) -> Axe
         ax.set_ylim(-1.0, 1.0)
     mean = np.mean(data)
     c0 = np.sum((data - mean) ** 2) / n
+
+    if c0 == 0:
+        raise ValueError(
+            "Cannot compute autocorrelation for a constant series (zero variance)"
+        )

     def r(h):
         return ((data[: n - h] - mean) * (data[h:] - mean)).sum() / n / c0
```