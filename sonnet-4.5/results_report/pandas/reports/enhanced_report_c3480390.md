# Bug Report: pandas.plotting.lag_plot Missing Input Validation for Lag Parameter

**Target**: `pandas.plotting.lag_plot`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `lag_plot()` function fails to validate its `lag` parameter, accepting zero and negative values that produce mathematically nonsensical results or confusing error messages, violating the expected behavior of a lag plot visualization.

## Property-Based Test

```python
import matplotlib
matplotlib.use('Agg')

import pandas as pd
from hypothesis import given, strategies as st, settings
import pandas.plotting as plotting


@given(
    data=st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10), min_size=2, max_size=100),
    lag=st.integers(min_value=-10, max_value=0)
)
@settings(max_examples=100)
def test_lag_plot_rejects_invalid_lag(data, lag):
    series = pd.Series(data)

    try:
        result = plotting.lag_plot(series, lag=lag)
        assert False, f"lag_plot should reject lag={lag} but it succeeded"
    except ValueError:
        pass

# Run the test
test_lag_plot_rejects_invalid_lag()
```

<details>

<summary>
**Failing input**: `lag=-1, data=[0.0, 0.0]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/19/hypo.py", line 24, in <module>
    test_lag_plot_rejects_invalid_lag()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/19/hypo.py", line 10, in test_lag_plot_rejects_invalid_lag
    data=st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10), min_size=2, max_size=100),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/19/hypo.py", line 19, in test_lag_plot_rejects_invalid_lag
    assert False, f"lag_plot should reject lag={lag} but it succeeded"
           ^^^^^
AssertionError: lag_plot should reject lag=-1 but it succeeded
Falsifying example: test_lag_plot_rejects_invalid_lag(
    data=[0.0, 0.0],  # or any other generated value
    lag=-1,
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/miniconda/lib/python3.13/site-packages/matplotlib/artist.py:211
        /home/npc/miniconda/lib/python3.13/site-packages/matplotlib/artist.py:212
        /home/npc/miniconda/lib/python3.13/site-packages/matplotlib/artist.py:276
        /home/npc/miniconda/lib/python3.13/site-packages/matplotlib/artist.py:288
        /home/npc/miniconda/lib/python3.13/site-packages/matplotlib/artist.py:446
        (and 94 more with settings.verbosity >= verbose)
```
</details>

## Reproducing the Bug

```python
import matplotlib
matplotlib.use('Agg')

import pandas as pd
import pandas.plotting as plotting

series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])

print("Test with lag=-1:")
result = plotting.lag_plot(series, lag=-1)
collections = result.collections[0]
offsets = collections.get_offsets()
print(f"Created {offsets.shape[0]} point(s) (should have raised ValueError)")
print(f"Y-axis label: '{result.get_ylabel()}'")
print(f"X-axis label: '{result.get_xlabel()}'")
print(f"Point coordinates: {offsets}")

print("\nTest with lag=-2:")
result = plotting.lag_plot(series, lag=-2)
collections = result.collections[0]
offsets = collections.get_offsets()
print(f"Created {offsets.shape[0]} point(s) (should have raised ValueError)")
print(f"Y-axis label: '{result.get_ylabel()}'")

print("\nTest with lag=0:")
try:
    result = plotting.lag_plot(series, lag=0)
    print("lag=0 succeeded unexpectedly")
except Exception as e:
    print(f"Raised {type(e).__name__}: {e}")
```

<details>

<summary>
Output showing invalid behavior
</summary>
```
Test with lag=-1:
Created 1 point(s) (should have raised ValueError)
Y-axis label: 'y(t + -1)'
X-axis label: 'y(t)'
Point coordinates: [[1.0 5.0]]

Test with lag=-2:
Created 1 point(s) (should have raised ValueError)
Y-axis label: 'y(t + -2)'

Test with lag=0:
Raised ValueError: x and y must be the same size
```
</details>

## Why This Is A Bug

1. **Mathematical invalidity**: Lag plots visualize the relationship between y(t) and y(t+k) where k > 0. The concept of lag in time series analysis is inherently positive - you compare each value with a value k steps ahead in time. Zero or negative lag values have no valid interpretation in this context.

2. **Nonsensical output for negative lag**: When lag=-1, the implementation uses array slicing that produces:
   - `y1 = data[:-(-1)]` becomes `data[:1]` (only the first element)
   - `y2 = data[-1:]` (only the last element)
   - This creates a single-point scatter plot, which provides no meaningful autocorrelation information

3. **Confusing axis labels**: For negative lag values, the Y-axis label becomes "y(t + -1)", which is mathematically awkward notation that would typically be written as "y(t - 1)" if it had meaning (but it doesn't in this context).

4. **Misleading error for lag=0**: When lag=0, the function crashes with "ValueError: x and y must be the same size" from matplotlib's scatter method, which doesn't clearly indicate that the problem is an invalid lag parameter value.

5. **Violates API contract expectations**: While the documentation states "lag : int, default 1" without explicitly forbidding non-positive values, the default of 1, the name "lag length", and all examples using positive values strongly imply that only positive integers are valid.

## Relevant Context

The lag_plot function is implemented in `/home/npc/miniconda/lib/python3.13/site-packages/pandas/plotting/_matplotlib/misc.py` at lines 427-441. The core logic uses array slicing:

```python
data = series.values
y1 = data[:-lag]  # Line 434
y2 = data[lag:]   # Line 435
```

This slicing logic assumes positive lag values. With negative lag:
- `data[:-(-n)]` becomes `data[:n]` (first n elements)
- `data[-n:]` (last n elements)

With lag=0:
- `data[:-0]` becomes `data[:0]` (empty array)
- `data[0:]` (full array)

Documentation reference: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.plotting.lag_plot.html

## Proposed Fix

```diff
--- a/pandas/plotting/_matplotlib/misc.py
+++ b/pandas/plotting/_matplotlib/misc.py
@@ -427,6 +427,9 @@ def lag_plot(series: Series, lag: int = 1, ax: Axes | None = None, **kwds) -> A
     # workaround because `c='b'` is hardcoded in matplotlib's scatter method
     import matplotlib.pyplot as plt

+    if lag < 1:
+        raise ValueError(f"lag must be a positive integer, got {lag}")
+
     kwds.setdefault("c", plt.rcParams["patch.facecolor"])

     data = series.values
```