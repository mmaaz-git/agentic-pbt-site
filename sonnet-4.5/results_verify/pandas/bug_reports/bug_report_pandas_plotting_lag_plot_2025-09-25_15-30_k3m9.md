# Bug Report: pandas.plotting.lag_plot Invalid Lag Parameter Validation

**Target**: `pandas.plotting.lag_plot`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`lag_plot()` accepts negative lag values without validation, producing nonsensical plots with incorrect axis labels and data. Additionally, `lag=0` crashes with an unclear error message from matplotlib instead of being properly validated and rejected.

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
```

**Failing input**: `lag=-1` (and any lag <= 0)

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
print(f"Created {offsets.shape[0]} point (should have raised ValueError)")
print(f"Y-axis label: '{result.get_ylabel()}'")

print("\nTest with lag=0:")
try:
    result = plotting.lag_plot(series, lag=0)
except Exception as e:
    print(f"Raised {type(e).__name__}: {e}")
```

**Output:**
```
Test with lag=-1:
Created 1 point (should have raised ValueError)
Y-axis label: 'y(t + -1)'

Test with lag=0:
Raised ValueError: x and y must be the same size
```

## Why This Is A Bug

1. **Violates API contract**: The docstring specifies "lag : int, default 1" without mentioning negative values. The default of 1 and the conceptual meaning of "lag length" strongly imply positive values only.

2. **Produces nonsensical results**: For `lag=-1`, the implementation does:
   - `y1 = data[:-(-1)] = data[:1]` (first element only)
   - `y2 = data[-1:]` (last element only)
   - Creates a single-point plot labeled "y(t + -1)" which is meaningless

3. **Unclear error message**: For `lag=0`, matplotlib raises "x and y must be the same size" which doesn't clearly indicate the problem is an invalid lag parameter.

4. **Lag plots are conceptually meaningless for lag ≤ 0**: A lag plot visualizes the relationship between `y(t)` and `y(t+k)` where `k > 0`. Zero or negative lag has no statistical interpretation in this context.

## Fix

```diff
diff --git a/pandas/plotting/_matplotlib/misc.py b/pandas/plotting/_matplotlib/misc.py
index 1234567..abcdefg 100644
--- a/pandas/plotting/_matplotlib/misc.py
+++ b/pandas/plotting/_matplotlib/misc.py
@@ -1,6 +1,9 @@
 def lag_plot(series: Series, lag: int = 1, ax: Axes | None = None, **kwds) -> Axes:
     # workaround because `c='b'` is hardcoded in matplotlib's scatter method
     import matplotlib.pyplot as plt
+
+    if lag < 1:
+        raise ValueError(f"lag must be a positive integer, got {lag}")

     kwds.setdefault("c", plt.rcParams["patch.facecolor"])