# Bug Report: pandas.plotting.parallel_coordinates Crashes with No Numeric Columns

**Target**: `pandas.plotting.parallel_coordinates`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `parallel_coordinates()` function crashes with `IndexError: list index out of range` when called with a DataFrame that has no numeric columns (only the class column), instead of validating the input or providing a clear error message.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pandas as pd
import pandas.plotting
import matplotlib.pyplot as plt

@given(st.integers(min_value=0, max_value=10))
def test_parallel_coordinates_handles_no_numeric_cols(n_rows):
    df = pd.DataFrame({'class': ['a'] * n_rows})
    fig, ax = plt.subplots()
    try:
        result = pandas.plotting.parallel_coordinates(df, 'class')
        assert result is not None
    except ValueError as e:
        assert "numeric" in str(e).lower() or "column" in str(e).lower()
    finally:
        plt.close(fig)
```

**Failing input**: DataFrame with only a class column, e.g., `pd.DataFrame({'class': ['a', 'b', 'c']})`

## Reproducing the Bug

```python
import pandas as pd
import pandas.plotting
import matplotlib.pyplot as plt

df = pd.DataFrame({'class': ['a', 'b', 'c']})
fig, ax = plt.subplots()

try:
    result = pandas.plotting.parallel_coordinates(df, 'class')
except IndexError as e:
    print(f"Error: {e}")

plt.close(fig)
```

**Output:**
```
Error: list index out of range
```

## Why This Is A Bug

The function does not validate that there is at least one numeric column remaining after removing the class column. The crash occurs when trying to set x-axis limits:

```python
ax.set_xlim(x[0], x[-1])
```

When the DataFrame has only the class column (or when `cols` is specified as an empty list), `x` becomes an empty list, causing an IndexError when trying to access `x[0]` or `x[-1]`.

The function should raise a clear ValueError indicating that at least one numeric column is required for parallel coordinates plotting.

## Fix

Add validation after determining the columns to plot:

```diff
def parallel_coordinates(...) -> Axes:
    ...
    if cols is None:
        df = frame.drop(class_column, axis=1)
    else:
        df = frame[cols]

    used_legends: set[str] = set()

    ncols = len(df.columns)
+
+   if ncols == 0:
+       raise ValueError(
+           "DataFrame must have at least one numeric column in addition to the class column"
+       )

    # determine values to use for xticks
    x: list[int] | Index
    if use_columns is True:
        ...
```