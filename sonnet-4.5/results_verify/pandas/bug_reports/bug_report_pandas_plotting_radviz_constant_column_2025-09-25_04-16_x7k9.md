# Bug Report: pandas.plotting.radviz Division by Zero on Constant Columns

**Target**: `pandas.plotting.radviz`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `radviz` function crashes with a ZeroDivisionError when any numeric column contains all identical values, due to division by zero in the normalization step.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from hypothesis.extra import pandas as hpd
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

@given(
    hpd.data_frames(
        columns=[
            hpd.column('A', elements=st.just(1.0)),
            hpd.column('B', elements=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False)),
            hpd.column('class', elements=st.sampled_from(['cat1', 'cat2']))
        ],
        index=hpd.range_indexes(min_size=2, max_size=10)
    )
)
def test_radviz_constant_column(df):
    fig, ax = plt.subplots()
    result = pd.plotting.radviz(df, 'class', ax=ax)
    plt.close('all')
```

**Failing input**: DataFrame with column `A` containing all identical values (e.g., all 1.0)

## Reproducing the Bug

```python
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

df = pd.DataFrame({
    'A': [1.0, 1.0, 1.0, 1.0],
    'B': [2.0, 3.0, 4.0, 5.0],
    'class': ['cat1', 'cat1', 'cat2', 'cat2']
})

fig, ax = plt.subplots()
result = pd.plotting.radviz(df, 'class', ax=ax)
```

## Why This Is A Bug

The `normalize` function in `radviz` (lines 147-150 of _matplotlib/misc.py) computes:

```python
def normalize(series):
    a = min(series)
    b = max(series)
    return (series - a) / (b - a)
```

For a constant column where all values are identical, `min(series) == max(series)`, so `b - a = 0`, causing `ZeroDivisionError`.

This is a valid input - users may have datasets with constant features, and the function should either:
1. Handle this gracefully (e.g., normalize to 0.5 or skip the column)
2. Provide a clear error message explaining the issue

## Fix

```diff
def normalize(series):
    a = min(series)
    b = max(series)
+   if a == b:
+       # Constant column - normalize to middle value
+       return series * 0 + 0.5
    return (series - a) / (b - a)
```

Alternatively, the function could validate inputs and provide a clear error:

```diff
def radviz(...):
+   # Validate that columns have variance
+   numeric_cols = frame.drop(class_column, axis=1)
+   for col in numeric_cols.columns:
+       if numeric_cols[col].min() == numeric_cols[col].max():
+           raise ValueError(
+               f"Column '{col}' has constant values. "
+               "RadViz requires columns with varying values."
+           )
    ...
```