# Bug Report: pandas.plotting.autocorrelation_plot Crashes with Empty Series

**Target**: `pandas.plotting.autocorrelation_plot`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `autocorrelation_plot()` function crashes with `ZeroDivisionError` when called with an empty Series, instead of validating the input or handling the edge case gracefully.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pandas as pd
import pandas.plotting
import matplotlib.pyplot as plt

@given(st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=0, max_size=100))
def test_autocorrelation_plot_handles_empty(data):
    series = pd.Series(data)
    fig, ax = plt.subplots()
    try:
        result = pandas.plotting.autocorrelation_plot(series)
        assert result is not None
    except ValueError as e:
        assert "empty" in str(e).lower() or "length" in str(e).lower()
    finally:
        plt.close(fig)
```

**Failing input**: `pd.Series([])`

## Reproducing the Bug

```python
import pandas as pd
import pandas.plotting
import matplotlib.pyplot as plt

empty_series = pd.Series([])
fig, ax = plt.subplots()

try:
    result = pandas.plotting.autocorrelation_plot(empty_series)
except ZeroDivisionError as e:
    print(f"Error: {e}")

plt.close(fig)
```

**Output:**
```
Error: division by zero
```

## Why This Is A Bug

The function does not validate that the input Series has at least one element before attempting to compute statistics. The crash occurs in the line:

```python
c0 = np.sum((data - mean) ** 2) / n
```

When `n=0`, this results in division by zero. The function should either:
1. Handle empty series gracefully (perhaps by returning an empty plot or a plot with a message)
2. Raise a clear ValueError indicating that the series must not be empty

The current ZeroDivisionError is confusing and doesn't indicate what the actual problem is.

## Fix

Add input validation at the beginning of the function:

```diff
def autocorrelation_plot(series: Series, ax: Axes | None = None, **kwds) -> Axes:
    import matplotlib.pyplot as plt

    n = len(series)
+   if n == 0:
+       raise ValueError("Series must contain at least one element for autocorrelation plot")
+
    data = np.asarray(series)
    if ax is None:
        ax = plt.gca()
        ax.set_xlim(1, n)
        ax.set_ylim(-1.0, 1.0)
    mean = np.mean(data)
    c0 = np.sum((data - mean) ** 2) / n
    ...
```