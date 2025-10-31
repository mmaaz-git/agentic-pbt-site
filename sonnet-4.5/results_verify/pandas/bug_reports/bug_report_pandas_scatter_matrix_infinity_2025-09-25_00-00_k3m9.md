# Bug Report: pandas.plotting.scatter_matrix crashes with infinity values

**Target**: `pandas.plotting.scatter_matrix`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`scatter_matrix` crashes with a `ValueError` when the DataFrame contains infinity values, even though DataFrames can legitimately contain such values and the function doesn't document this restriction.

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

**Failing input**: DataFrame with infinity values in any column

## Reproducing the Bug

```python
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')

df = pd.DataFrame({
    'A': [1.0, 2.0, 3.0],
    'B': [4.0, 5.0, 6.0],
    'C': [-np.inf, 7.0, 8.0]
})

pd.plotting.scatter_matrix(df)
```

Output:
```
ValueError: Axis limits cannot be NaN or Inf
```

## Why This Is A Bug

1. DataFrames can legitimately contain infinity values (e.g., from mathematical operations like division by zero)
2. The function signature and docstring don't document any restriction on infinity values
3. Other pandas plotting functions (like `df.plot()`) handle infinity values gracefully
4. The function should either handle infinity values properly or validate inputs and raise a clear error message

## Fix

The function should filter out infinity values when calculating axis boundaries, similar to how it already filters NaN values using `mask = notna(df)`:

```diff
--- a/pandas/plotting/_matplotlib/misc.py
+++ b/pandas/plotting/_matplotlib/misc.py
@@ -66,7 +66,8 @@ def scatter_matrix(

     boundaries_list = []
     for a in df.columns:
-        values = df[a].values[mask[a].values]
+        valid_mask = mask[a].values & np.isfinite(df[a].values)
+        values = df[a].values[valid_mask]
         rmin_, rmax_ = np.min(values), np.max(values)
         rdelta_ext = (rmax_ - rmin_) * range_padding / 2
         boundaries_list.append((rmin_ - rdelta_ext, rmax_ + rdelta_ext))
@@ -75,7 +76,8 @@ def scatter_matrix(
         for j, b in enumerate(df.columns):
             ax = axes[i, j]

             if i == j:
-                values = df[a].values[mask[a].values]
+                valid_mask = mask[a].values & np.isfinite(df[a].values)
+                values = df[a].values[valid_mask]

                 # Deal with the diagonal by drawing a histogram there.
                 if diagonal == "hist":
```