# Bug Report: pandas.plotting.lag_plot Negative Lag Parameter

**Target**: `pandas.plotting.lag_plot`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `lag_plot` function silently accepts negative lag values and produces meaningless plots instead of raising a clear validation error.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st
import pandas as pd
import pandas.plotting
import pytest

@settings(max_examples=200)
@given(
    data=st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=2, max_size=100),
    lag=st.integers(min_value=-10, max_value=-1)
)
def test_lag_plot_negative_lag(data, lag):
    series = pd.Series(data)

    with pytest.raises(ValueError):
        pandas.plotting.lag_plot(series, lag=lag)
```

**Failing input**: `pandas.plotting.lag_plot(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]), lag=-1)`

## Reproducing the Bug

```python
import pandas as pd
import pandas.plotting
import matplotlib
matplotlib.use('Agg')

series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])

result = pandas.plotting.lag_plot(series, lag=-1)
print(f"Function succeeded with lag=-1")
print(f"Returned: {result}")

data = series.values
y1 = data[:-(-1)]
y2 = data[-1:]
print(f"\nWhat gets plotted:")
print(f"y1 = data[:1] = {y1}")
print(f"y2 = data[-1:] = {y2}")
print(f"This plots only the first element vs the last element - not a lag plot!")
```

## Why This Is A Bug

1. **Semantically meaningless**: A lag plot visualizes the correlation between a time series and a lagged version of itself. Negative lag values have no meaningful interpretation in this context.

2. **Unexpected behavior**: Due to Python's array slicing semantics, `data[:-(-1)]` equals `data[:1]`, which returns only the first element. Combined with `data[-1:]` (last element), this creates a single-point "plot" that bears no resemblance to a lag plot.

3. **Silent failure**: The function should validate that lag is positive, but instead silently produces garbage output that users may not immediately recognize as incorrect.

4. **Inconsistent with documentation**: The function signature has `lag: int = 1` with default 1, implying positive values are expected. The parameter docstring says "Lag length of the scatter plot" without mentioning negative values are allowed.

## Fix

```diff
 def lag_plot(series: Series, lag: int = 1, ax: Axes | None = None, **kwds) -> Axes:
+    if lag <= 0:
+        raise ValueError(
+            f"lag must be a positive integer, got {lag}. "
+            f"Lag represents the offset between y(t) and y(t+lag)."
+        )
     import matplotlib.pyplot as plt

     kwds.setdefault("c", plt.rcParams["patch.facecolor"])
```