# Bug Report: dask.dataframe DataFrame.nlargest() Crashes When Followed by Column Selection

**Target**: `dask.dataframe.DataFrame.nlargest()`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

Calling `DataFrame.nlargest(n, column)[column].compute()` crashes with `TypeError: Series.nlargest() got an unexpected keyword argument 'columns'`. The lazy evaluation incorrectly tries to apply DataFrame.nlargest parameters to a Series operation.

## Property-Based Test

```python
import pandas as pd
import dask.dataframe as dd
from hypothesis import given, strategies as st, settings


@given(
    st.lists(
        st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=30
    )
)
@settings(max_examples=50)
def test_nlargest_column_selection(values):
    n = min(5, len(values))

    pdf = pd.DataFrame({'values': values})
    ddf = dd.from_pandas(pdf, npartitions=2)

    result = ddf.nlargest(n, 'values')['values'].compute()
    expected = pdf.nlargest(n, 'values')['values']

    assert len(result) == len(expected)
```

**Failing input**: Any DataFrame, e.g., `pd.DataFrame({'values': [1.0, 2.0, 3.0]})`

## Reproducing the Bug

```python
import pandas as pd
import dask.dataframe as dd

pdf = pd.DataFrame({'values': [1.0, 2.0, 3.0]})
ddf = dd.from_pandas(pdf, npartitions=2)

try:
    result = ddf.nlargest(2, 'values')['values'].compute()
    print("Success:", result)
except TypeError as e:
    print(f"Error: {e}")
```

Output:
```
Error: Series.nlargest() got an unexpected keyword argument 'columns'
```

## Why This Is A Bug

This pattern works fine in pandas:
```python
pdf.nlargest(2, 'values')['values']  # Returns a Series
```

The dask lazy evaluation system should handle the same pattern, but it incorrectly propagates the DataFrame.nlargest() parameters (`columns='values'`) when the intermediate result becomes a Series after the column selection.

**Workaround**: Compute before selecting the column:
```python
ddf.nlargest(2, 'values').compute()['values']  # Works
```

or use Series.nlargest directly:
```python
ddf['values'].nlargest(2).compute()  # Works
```

## Fix

The bug appears to be in the lazy evaluation / query optimization layer. When `df.nlargest(n, col)[col]` is evaluated:
1. `df.nlargest(n, col)` creates a lazy DataFrame operation with parameters `n` and `columns=col`
2. `[col]` selects a single column, which should produce a Series
3. The bug: dask tries to evaluate the nlargest operation on the Series but still passes the `columns` parameter

The fix would need to detect when a DataFrame operation is followed by column selection that produces a Series, and adjust the operation parameters accordingly.