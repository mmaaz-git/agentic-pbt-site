# Bug Report: pandas.plotting.scatter_matrix crashes with single-row DataFrames

**Target**: `pandas.plotting.scatter_matrix`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`scatter_matrix` crashes with a `ValueError` when given a single-row DataFrame with certain numeric values, due to matplotlib's histogram being unable to create bins for a single unique value.

## Property-Based Test

```python
from hypothesis import given, assume, settings
from hypothesis.extra.pandas import column, data_frames, range_indexes
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

@settings(max_examples=200, deadline=None)
@given(
    df=data_frames(
        columns=[
            column('A', dtype=float),
            column('B', dtype=float),
            column('C', dtype=float),
        ],
        index=range_indexes(min_size=1, max_size=50)
    )
)
def test_scatter_matrix_shape_property(df):
    assume(not df.empty)
    assume(not df.isna().all().all())

    result = pd.plotting.scatter_matrix(df)
    n_cols = len(df.columns)
    assert result.shape == (n_cols, n_cols)
    plt.close('all')
```

**Failing input**: Single-row DataFrame with certain numeric values

## Reproducing the Bug

```python
import pandas as pd
import matplotlib
matplotlib.use('Agg')

df = pd.DataFrame({
    'A': [0.0],
    'B': [-1.297501e+16],
    'C': [-1.297501e+16]
})

pd.plotting.scatter_matrix(df)
```

Output:
```
ValueError: Too many bins for data range. Cannot create 10 finite-sized bins.
```

## Why This Is A Bug

1. Single-row DataFrames are valid pandas DataFrames
2. The function should either handle this edge case or document a minimum row requirement
3. The crash occurs in matplotlib's histogram when trying to create bins for a single unique value
4. The function works fine for some single-row DataFrames but crashes for others depending on the numeric values

## Fix

Add validation for minimum data points or handle the histogram creation more gracefully:

```diff
--- a/pandas/plotting/_matplotlib/misc.py
+++ b/pandas/plotting/_matplotlib/misc.py
@@ -56,6 +56,11 @@ def scatter_matrix(
     **kwds,
 ):
     df = frame._get_numeric_data()
+
+    if len(df) < 2:
+        raise ValueError(
+            "scatter_matrix requires at least 2 rows of data"
+        )
+
     n = df.columns.size
     naxes = n * n
     fig, axes = create_subplots(naxes=naxes, figsize=figsize, ax=ax, squeeze=False)
```

Alternatively, the function could fall back to a simpler plot when there's insufficient data for histograms.