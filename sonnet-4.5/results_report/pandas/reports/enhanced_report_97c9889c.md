# Bug Report: pandas.plotting.autocorrelation_plot Produces NaN Values for Constant Series Without Proper Error Handling

**Target**: `pandas.plotting.autocorrelation_plot`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `autocorrelation_plot` function silently produces a plot filled entirely with NaN values when given a constant time series (zero variance), issuing only a RuntimeWarning about division by zero instead of raising a proper error or handling the special case gracefully.

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
        assert not np.all(np.isnan(ydata)), f"All autocorrelation values are NaN for constant series with value={value}, length={length}"

    plt.close('all')

if __name__ == "__main__":
    # Run the test
    test_autocorrelation_plot_constant_series()
```

<details>

<summary>
**Failing input**: `value=0.0, length=2`
</summary>
```
/home/npc/miniconda/lib/python3.13/site-packages/pandas/plotting/_matplotlib/misc.py:457: RuntimeWarning: invalid value encountered in scalar divide
  return ((data[: n - h] - mean) * (data[h:] - mean)).sum() / n / c0
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/2/hypo.py", line 26, in <module>
    test_autocorrelation_plot_constant_series()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/2/hypo.py", line 9, in test_autocorrelation_plot_constant_series
    @given(

  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/2/hypo.py", line 20, in test_autocorrelation_plot_constant_series
    assert not np.all(np.isnan(ydata)), f"All autocorrelation values are NaN for constant series with value={value}, length={length}"
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: All autocorrelation values are NaN for constant series with value=0.0, length=2
Falsifying example: test_autocorrelation_plot_constant_series(
    # The test sometimes passed when commented parts were varied together.
    value=0.0,  # or any other generated value
    length=2,  # or any other generated value
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/miniconda/lib/python3.13/site-packages/matplotlib/cbook.py:1741
```
</details>

## Reproducing the Bug

```python
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Create a constant series
series = pd.Series([42.0] * 20)
print(f"Input series: {series.tolist()[:5]}... (all values are 42.0)")
print(f"Series variance: {series.var()}")

# Call autocorrelation_plot
result = pd.plotting.autocorrelation_plot(series)

# Extract the autocorrelation values from the plot
lines = result.get_lines()
autocorr_line = lines[-1]  # The last line contains the autocorrelation values
ydata = autocorr_line.get_ydata()

print(f"\nAutocorrelation values (first 5): {ydata[:5]}")
print(f"All autocorrelation values are NaN: {np.all(np.isnan(ydata))}")
print(f"Number of NaN values: {np.sum(np.isnan(ydata))} out of {len(ydata)}")

plt.close('all')
```

<details>

<summary>
RuntimeWarning and NaN output for constant series
</summary>
```
/home/npc/miniconda/lib/python3.13/site-packages/pandas/plotting/_matplotlib/misc.py:457: RuntimeWarning: invalid value encountered in scalar divide
  return ((data[: n - h] - mean) * (data[h:] - mean)).sum() / n / c0
Input series: [42.0, 42.0, 42.0, 42.0, 42.0]... (all values are 42.0)
Series variance: 0.0

Autocorrelation values (first 5): [nan nan nan nan nan]
All autocorrelation values are NaN: True
Number of NaN values: 20 out of 20
```
</details>

## Why This Is A Bug

This violates expected behavior for several critical reasons:

1. **Mathematically Undefined Operation**: Autocorrelation is mathematically undefined for constant series (zero variance), as confirmed by standard mathematical definitions. The function attempts division by zero when calculating `c0 = sum((data - mean)**2) / n`, which equals 0 for constant series.

2. **Silent Failure**: The function produces a completely meaningless plot filled with NaN values without raising a proper exception. Users receive only a RuntimeWarning that can be easily missed, especially in production environments or automated pipelines.

3. **Documentation Gap**: The function's documentation (available at https://pandas.pydata.org/docs/reference/api/pandas.plotting.autocorrelation_plot.html) makes no mention of this limitation or special case handling for constant series.

4. **Inconsistent with Pandas Conventions**: Other pandas functions typically raise ValueError for mathematically undefined operations rather than returning NaN-filled results. For example, correlation functions raise errors for constant inputs.

5. **Real-World Impact**: Constant series occur in real datasets (e.g., monitoring systems during steady-state periods, financial instruments with no trading activity, sensor data when values are stuck). Users expect either a meaningful error or special handling.

## Relevant Context

The issue occurs in `/pandas/plotting/_matplotlib/misc.py` at lines 454-457:
- Line 454: `c0 = np.sum((data - mean) ** 2) / n` computes the variance (which is 0 for constant series)
- Line 457: Division by `c0` in the autocorrelation calculation causes the NaN values

The function already sets up the plot axes and draws confidence bands before discovering the data is constant, resulting in a visually complete but mathematically meaningless plot.

Related pandas issue discussions suggest that proper error handling for edge cases is a priority for the library's user experience.

## Proposed Fix

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
+            "Cannot compute autocorrelation for a constant series (variance is zero). "
+            "Autocorrelation is mathematically undefined when all values are identical."
+        )

     def r(h):
         return ((data[: n - h] - mean) * (data[h:] - mean)).sum() / n / c0
```