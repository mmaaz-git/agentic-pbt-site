# Bug Report: pandas.plotting.lag_plot Accepts Invalid Negative Lag Values

**Target**: `pandas.plotting.lag_plot`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `lag_plot` function silently accepts negative lag values and produces mathematically meaningless single-point plots instead of raising a validation error.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st
import pandas as pd
import pandas.plotting
import pytest
import matplotlib
matplotlib.use('Agg')

@settings(max_examples=200)
@given(
    data=st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=2, max_size=100),
    lag=st.integers(min_value=-10, max_value=-1)
)
def test_lag_plot_negative_lag(data, lag):
    series = pd.Series(data)

    with pytest.raises(ValueError):
        pandas.plotting.lag_plot(series, lag=lag)

if __name__ == "__main__":
    # Run the test to find failing examples
    test_lag_plot_negative_lag()
```

<details>

<summary>
**Failing input**: `pandas.plotting.lag_plot(pd.Series([0.0, 0.0]), lag=-1)`
</summary>
```
/home/npc/miniconda/lib/python3.13/site-packages/matplotlib/axes/_base.py:3060: RuntimeWarning: overflow encountered in scalar add
  x0, x1 = inverse_trans.transform([x0t - delta, x1t + delta])
/home/npc/miniconda/lib/python3.13/site-packages/matplotlib/axes/_base.py:3057: RuntimeWarning: overflow encountered in scalar subtract
  delta = (x1t - x0t) * margin
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/63/hypo.py", line 21, in <module>
    test_lag_plot_negative_lag()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/63/hypo.py", line 9, in test_lag_plot_negative_lag
    @given(

  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/63/hypo.py", line 16, in test_lag_plot_negative_lag
    with pytest.raises(ValueError):
         ~~~~~~~~~~~~~^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/_pytest/raises.py", line 712, in __exit__
    fail(f"DID NOT RAISE {self.expected_exceptions[0]!r}")
    ~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/_pytest/outcomes.py", line 177, in fail
    raise Failed(msg=reason, pytrace=pytrace)
Failed: DID NOT RAISE <class 'ValueError'>
Falsifying example: test_lag_plot_negative_lag(
    # The test always failed when commented parts were varied together.
    data=[0.0, 0.0],  # or any other generated value
    lag=-1,  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
import pandas as pd
import pandas.plotting
import matplotlib
matplotlib.use('Agg')

# Create a simple time series
series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])

# Try to create a lag plot with negative lag
result = pandas.plotting.lag_plot(series, lag=-1)
print(f"Function succeeded with lag=-1")
print(f"Returned: {result}")

# Show what actually gets plotted
data = series.values
y1 = data[:-(-1)]  # This becomes data[:1]
y2 = data[-1:]      # This is data[-1:]
print(f"\nWhat gets plotted:")
print(f"y1 (x-axis) = data[:-(-1)] = data[:1] = {y1}")
print(f"y2 (y-axis) = data[-1:] = data[-1:] = {y2}")
print(f"Number of points plotted: {len(y1)} x {len(y2)} = {min(len(y1), len(y2))} points")
print(f"\nThis plots only the first element ({y1[0]}) vs the last element ({y2[0]}) - not a lag plot!")

# Show what the axis labels look like
print(f"\nAxis labels:")
print(f"x-axis: y(t)")
print(f"y-axis: y(t + {-1})")  # This would show as "y(t + -1)" which is confusing
```

<details>

<summary>
Function executes without error but produces incorrect single-point plot
</summary>
```
Function succeeded with lag=-1
Returned: Axes(0.125,0.11;0.775x0.77)

What gets plotted:
y1 (x-axis) = data[:-(-1)] = data[:1] = [1.]
y2 (y-axis) = data[-1:] = data[-1:] = [5.]
Number of points plotted: 1 x 1 = 1 points

This plots only the first element (1.0) vs the last element (5.0) - not a lag plot!

Axis labels:
x-axis: y(t)
y-axis: y(t + -1)
```
</details>

## Why This Is A Bug

This violates expected behavior in several critical ways:

1. **Mathematical invalidity**: Lag plots are defined to show the relationship between a time series at time t and time t-lag (i.e., lag periods in the past). A negative lag value has no valid interpretation in autocorrelation analysis, which is the primary purpose of lag plots.

2. **Incorrect array slicing behavior**: Due to Python's negative indexing, `data[:-(-1)]` becomes `data[:1]` (just the first element) and `data[-1:]` gives the last element. This creates a meaningless single-point "plot" instead of the expected scatter plot showing autocorrelation patterns.

3. **Misleading axis labels**: The y-axis label becomes "y(t + -1)" which is confusing and mathematically incorrect notation. The standard interpretation would be y(t-1), but that's not what's being plotted.

4. **Silent failure without validation**: The function accepts invalid input without any warning or error, violating the principle of failing fast and clearly. Users may not immediately realize their analysis is based on garbage output.

5. **Documentation inconsistency**: While the documentation doesn't explicitly prohibit negative values, the default value is 1 (positive), all examples use positive values, and the statistical definition of lag plots requires positive lag values.

## Relevant Context

The bug occurs in the matplotlib backend implementation located in `pandas.plotting._matplotlib.misc`. The core issue is in these lines:

```python
def lag_plot(series: Series, lag: int = 1, ax: Axes | None = None, **kwds) -> Axes:
    # ... setup code ...
    data = series.values
    y1 = data[:-lag]  # With lag=-1, this becomes data[:1]
    y2 = data[lag:]    # With lag=-1, this becomes data[-1:]
```

Statistical references:
- NIST Engineering Statistics Handbook defines lag plots as plotting Yi versus Yi-k where k is a positive integer
- Standard time series analysis texts (e.g., Box & Jenkins) always use positive lags for autocorrelation analysis
- The pandas documentation at https://pandas.pydata.org/docs/reference/api/pandas.plotting.lag_plot.html shows only positive lag examples

Related pandas functions like `Series.autocorr()` and `Series.shift()` handle lag parameters differently, with `shift()` allowing negative values for leading (future) values but clearly documenting this behavior.

## Proposed Fix

```diff
 def lag_plot(series: Series, lag: int = 1, ax: Axes | None = None, **kwds) -> Axes:
+    if lag <= 0:
+        raise ValueError(
+            f"lag must be a positive integer, got {lag}. "
+            f"Lag plots visualize autocorrelation by plotting y(t) vs y(t-lag)."
+        )
     # workaround because `c='b'` is hardcoded in matplotlib's scatter method
     import matplotlib.pyplot as plt

     kwds.setdefault("c", plt.rcParams["patch.facecolor"])

     data = series.values
     y1 = data[:-lag]
     y2 = data[lag:]
```