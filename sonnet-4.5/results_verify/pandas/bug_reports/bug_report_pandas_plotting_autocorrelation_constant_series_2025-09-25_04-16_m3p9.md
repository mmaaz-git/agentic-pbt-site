# Bug Report: pandas.plotting.autocorrelation_plot Division by Zero on Constant Series

**Target**: `pandas.plotting.autocorrelation_plot`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `autocorrelation_plot` function crashes with a ZeroDivisionError (or produces NaN/Inf values with warnings) when the input series contains all identical values, due to division by zero in the autocorrelation calculation.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from hypothesis.extra import pandas as hpd
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

@given(
    hpd.series(
        elements=st.just(5.0),
        index=hpd.range_indexes(min_size=10, max_size=50)
    )
)
def test_autocorrelation_constant_series(series):
    fig, ax = plt.subplots()
    result = pd.plotting.autocorrelation_plot(series, ax=ax)
    plt.close('all')
```

**Failing input**: Series with all identical values (e.g., `pd.Series([5.0] * 20)`)

## Reproducing the Bug

```python
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

series = pd.Series([5.0] * 20)

fig, ax = plt.subplots()
result = pd.plotting.autocorrelation_plot(series, ax=ax)
```

## Why This Is A Bug

The `autocorrelation_plot` function (lines 444-474 of _matplotlib/misc.py) computes:

```python
mean = np.mean(data)
c0 = np.sum((data - mean) ** 2) / n

def r(h):
    return ((data[: n - h] - mean) * (data[h:] - mean)).sum() / n / c0
```

For a constant series:
- All values equal the mean: `data[i] == mean` for all i
- Therefore: `(data - mean) ** 2 = 0` for all elements
- Thus: `c0 = 0`
- The function `r(h)` then divides by `c0`, causing `ZeroDivisionError` or producing `nan`/`inf` values

This is a legitimate edge case - users may want to understand why a time series has no autocorrelation (because it's constant), and the function should handle this gracefully.

## Fix

```diff
def autocorrelation_plot(series, ax=None, **kwds):
    import matplotlib.pyplot as plt

    n = len(series)
    data = np.asarray(series)
    if ax is None:
        ax = plt.gca()
        ax.set_xlim(1, n)
        ax.set_ylim(-1.0, 1.0)
    mean = np.mean(data)
    c0 = np.sum((data - mean) ** 2) / n

+   # Handle constant series (no variance)
+   if c0 == 0:
+       # Constant series has undefined autocorrelation
+       # Plot zeros to indicate no autocorrelation structure
+       x = np.arange(n) + 1
+       y = np.zeros(n)
+       ax.plot(x, y, **kwds)
+       ax.axhline(y=0.0, color="black")
+       ax.set_xlabel("Lag")
+       ax.set_ylabel("Autocorrelation")
+       if "label" in kwds:
+           ax.legend()
+       ax.grid()
+       return ax

    def r(h):
        return ((data[: n - h] - mean) * (data[h:] - mean)).sum() / n / c0
    ...
```