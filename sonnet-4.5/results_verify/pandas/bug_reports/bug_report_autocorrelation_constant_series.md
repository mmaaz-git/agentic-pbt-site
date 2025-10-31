# Bug Report: pandas.plotting.autocorrelation_plot Division by Zero

**Target**: `pandas.plotting.autocorrelation_plot`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `autocorrelation_plot` function crashes with a division by zero error when given a Series where all values are identical (constant series), instead of handling this edge case gracefully.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import hypothesis.extra.pandas as hpd
import pandas.plotting

@given(
    value=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
    size=st.integers(min_value=2, max_value=100)
)
def test_autocorrelation_constant_series(value, size):
    data = pd.Series([value] * size)
    result = pandas.plotting.autocorrelation_plot(data)
```

**Failing input**: `pd.Series([5.0, 5.0, 5.0, 5.0, 5.0])`

## Reproducing the Bug

```python
import pandas as pd
import pandas.plotting
import matplotlib
matplotlib.use('Agg')

constant_series = pd.Series([5.0, 5.0, 5.0, 5.0, 5.0])

try:
    result = pandas.plotting.autocorrelation_plot(constant_series)
except Exception as e:
    print(f"Error: {e}")
```

## Why This Is A Bug

Constant series (where all values are identical) are valid input for time series analysis. Users might encounter such data when:
1. A sensor reports a constant value
2. A variable remains unchanged over a time period
3. Testing or debugging code with simple data

The autocorrelation of a constant series is mathematically undefined (variance is zero), but the function should handle this gracefully with either:
1. A clear error message explaining the issue
2. Return NaN values for the autocorrelation
3. Display an informative plot indicating no variance

Instead, the function crashes with a cryptic division error.

## Fix

The bug occurs in `pandas/plotting/_matplotlib/misc.py` at line 454-457:

```python
c0 = np.sum((data - mean) ** 2) / n

def r(h):
    return ((data[: n - h] - mean) * (data[h:] - mean)).sum() / n / c0
```

When all values are equal, `c0` becomes 0, causing division by zero in `r(h)`.

Proposed fix:

```diff
def autocorrelation_plot(series: Series, ax: Axes | None = None, **kwds) -> Axes:
    import matplotlib.pyplot as plt

    n = len(series)
    data = np.asarray(series)
    if ax is None:
        ax = plt.gca()
        ax.set_xlim(1, n)
        ax.set_ylim(-1.0, 1.0)
    mean = np.mean(data)
    c0 = np.sum((data - mean) ** 2) / n

+    if c0 == 0:
+        import warnings
+        warnings.warn(
+            "Input series has zero variance. Autocorrelation is undefined. "
+            "Returning plot with all autocorrelations set to NaN.",
+            UserWarning
+        )
+        x = np.arange(n) + 1
+        y = np.full(n, np.nan)
+    else:
+        def r(h):
+            return ((data[: n - h] - mean) * (data[h:] - mean)).sum() / n / c0
+
+        x = np.arange(n) + 1
+        y = [r(loc) for loc in x]

-    def r(h):
-        return ((data[: n - h] - mean) * (data[h:] - mean)).sum() / n / c0
-
-    x = np.arange(n) + 1
-    y = [r(loc) for loc in x]
    z95 = 1.959963984540054
    z99 = 2.5758293035489004
    ax.axhline(y=z99 / np.sqrt(n), linestyle="--", color="grey")
    ax.axhline(y=z95 / np.sqrt(n), color="grey")
    ax.axhline(y=0.0, color="black")
    ax.axhline(y=-z95 / np.sqrt(n), color="grey")
    ax.axhline(y=-z99 / np.sqrt(n), linestyle="--", color="grey")
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    ax.plot(x, y, **kwds)
    if "label" in kwds:
        ax.legend()
    ax.grid()
    return ax
```