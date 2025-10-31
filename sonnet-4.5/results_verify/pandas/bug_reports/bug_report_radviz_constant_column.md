# Bug Report: pandas.plotting.radviz Division by Zero with Constant Column

**Target**: `pandas.plotting.radviz`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `radviz` function crashes with a division by zero error when the input DataFrame contains a column where all values are identical (constant column), instead of handling this edge case gracefully.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pandas as pd
import pandas.plotting

@given(
    constant_value=st.floats(allow_nan=False, allow_infinity=False),
    size=st.integers(min_value=2, max_value=20)
)
def test_radviz_constant_column(constant_value, size):
    df = pd.DataFrame({
        'A': list(range(size)),
        'B': [constant_value] * size,
        'class': ['c1', 'c2'] * (size // 2) + ['c1'] * (size % 2)
    })
    result = pandas.plotting.radviz(df, 'class')
```

**Failing input**:
```python
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [5.0, 5.0, 5.0],
    'class': ['c1', 'c2', 'c1']
})
pandas.plotting.radviz(df, 'class')
```

## Reproducing the Bug

```python
import pandas as pd
import pandas.plotting
import matplotlib
matplotlib.use('Agg')

df = pd.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': [5.0, 5.0, 5.0, 5.0, 5.0],
    'class': ['c1', 'c2', 'c1', 'c2', 'c1']
})

try:
    result = pandas.plotting.radviz(df, 'class')
except Exception as e:
    print(f"Error: {e}")
```

## Why This Is A Bug

DataFrames with constant columns (where all values in a column are identical) are valid input for multivariate visualization. Users might encounter such data when:
1. A feature has no variance in the dataset
2. A measurement stays constant across all observations
3. Testing or debugging with simple data

The radviz visualization should handle this gracefully with either:
1. A clear error message explaining that the column has no variance
2. Skip normalization for constant columns (treat as 0.5)
3. Display an informative message to the user

Instead, the function crashes with a cryptic division error.

## Fix

The bug occurs in `pandas/plotting/_matplotlib/misc.py` at lines 147-150:

```python
def normalize(series):
    a = min(series)
    b = max(series)
    return (series - a) / (b - a)
```

When all values in a series are equal, `a == b`, so `(b - a) == 0`, causing division by zero.

Proposed fix:

```diff
def radviz(
    frame: DataFrame,
    class_column,
    ax: Axes | None = None,
    color=None,
    colormap=None,
    **kwds,
) -> Axes:
    import matplotlib.pyplot as plt

    def normalize(series):
        a = min(series)
        b = max(series)
+        if a == b:
+            return series * 0
        return (series - a) / (b - a)

    n = len(frame)
    classes = frame[class_column].drop_duplicates()
    class_col = frame[class_column]
    df = frame.drop(class_column, axis=1).apply(normalize)
    ...
```

Alternatively, for better user experience:

```diff
def radviz(
    frame: DataFrame,
    class_column,
    ax: Axes | None = None,
    color=None,
    colormap=None,
    **kwds,
) -> Axes:
    import matplotlib.pyplot as plt

    def normalize(series):
        a = min(series)
        b = max(series)
+        if a == b:
+            import warnings
+            warnings.warn(
+                f"Column '{series.name}' has zero variance (all values are {a}). "
+                f"It will be treated as having normalized value 0.",
+                UserWarning
+            )
+            return series * 0
        return (series - a) / (b - a)

    n = len(frame)
    classes = frame[class_column].drop_duplicates()
    class_col = frame[class_column]
    df = frame.drop(class_column, axis=1).apply(normalize)
    ...
```