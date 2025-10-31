# Bug Report: pandas.plotting.autocorrelation_plot Division by Zero with Constant Series

**Target**: `pandas.plotting.autocorrelation_plot`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `autocorrelation_plot` function triggers a RuntimeWarning for division by zero when processing a constant series (where all values are identical), resulting in NaN values in the autocorrelation computation.

## Property-Based Test

```python
@given(
    constant_value=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
    length=st.integers(min_value=10, max_value=50)
)
@settings(max_examples=50)
def test_any_constant_series(self, constant_value, length):
    """
    Property: autocorrelation_plot should handle any constant series without warnings.
    """
    series = pd.Series([constant_value] * length)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", RuntimeWarning)
        result = pandas.plotting.autocorrelation_plot(series)
        plt.close('all')

        runtime_warnings = [warning for warning in w if issubclass(warning.category, RuntimeWarning)]
        assert len(runtime_warnings) == 0, \
            f"autocorrelation_plot should not raise warnings for constant={constant_value}"
```

**Failing input**: Any constant series, e.g., `pd.Series([5.0] * 20)` or `pd.Series([0.0] * 20)`

## Reproducing the Bug

```python
import pandas as pd
import pandas.plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings

constant_series = pd.Series([5.0] * 20)

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always", RuntimeWarning)
    result = pandas.plotting.autocorrelation_plot(constant_series)
    plt.close('all')

    runtime_warnings = [warning for warning in w if issubclass(warning.category, RuntimeWarning)]
    if runtime_warnings:
        print(f"RuntimeWarning: {runtime_warnings[0].message}")
```

Output:
```
RuntimeWarning: invalid value encountered in scalar divide
```

## Why This Is A Bug

The bug occurs in `/pandas/plotting/_matplotlib/misc.py` in the `autocorrelation_plot` function:

```python
def autocorrelation_plot(series: Series, ax: Axes | None = None, **kwds) -> Axes:
    n = len(series)
    data = np.asarray(series)
    mean = np.mean(data)
    c0 = np.sum((data - mean) ** 2) / n  # This is the variance

    def r(h):
        return ((data[: n - h] - mean) * (data[h:] - mean)).sum() / n / c0  # Division by c0

    # ... rest of function
```

For a constant series where all values are identical:
1. `mean` equals the constant value
2. `(data - mean)` is all zeros
3. `c0 = np.sum(0**2) / n = 0` (variance is zero)
4. In the `r(h)` function, division by `c0` causes division by zero
5. This results in NaN values and a RuntimeWarning

This violates the expected behavior that plotting functions should handle all valid numeric input gracefully. A constant series is mathematically valid input that should either:
- Plot correctly (with autocorrelation values defined appropriately), or
- Raise a clear, informative error message

## Fix

Add a check for zero variance and handle it appropriately:

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

+   # Handle constant series (zero variance)
+   if c0 == 0:
+       # For a constant series, autocorrelation is undefined
+       # We'll plot NaN values without the division warning
+       x = np.arange(n) + 1
+       y = [np.nan] * n
+   else:
-       def r(h):
-           return ((data[: n - h] - mean) * (data[h:] - mean)).sum() / n / c0
+       def r(h):
+           return ((data[: n - h] - mean) * (data[h:] - mean)).sum() / n / c0

-       x = np.arange(n) + 1
-       y = [r(loc) for loc in x]
+       x = np.arange(n) + 1
+       y = [r(loc) for loc in x]

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

Alternatively, raise a clear error:

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

+   # Handle constant series (zero variance)
+   if c0 == 0:
+       raise ValueError(
+           "Cannot compute autocorrelation for a constant series "
+           "(all values are identical, variance is zero). "
+           "Autocorrelation is only defined for series with non-zero variance."
+       )

    def r(h):
        return ((data[: n - h] - mean) * (data[h:] - mean)).sum() / n / c0

    # ... rest of function
```