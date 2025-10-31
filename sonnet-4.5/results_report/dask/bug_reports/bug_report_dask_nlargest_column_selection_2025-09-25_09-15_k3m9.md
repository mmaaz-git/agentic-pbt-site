# Bug Report: dask.dataframe.dask_expr NLargest Column Selection

**Target**: `dask.dataframe.dask_expr._reductions.NLargest`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

Calling `.compute()` on the result of `df.nlargest(n, column)[column]` crashes with `TypeError: Series.nlargest() got an unexpected keyword argument 'columns'`. This occurs because when a column is selected from a DataFrame.nlargest() result, dask incorrectly passes the DataFrame's `columns` parameter to the underlying pandas Series.nlargest() method, which doesn't accept that argument.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings, assume
import pandas as pd
import dask.dataframe as dd


@settings(max_examples=50)
@given(
    data=st.lists(st.integers(min_value=-100, max_value=100), min_size=5, max_size=20),
    n=st.integers(min_value=1, max_value=5)
)
def test_nlargest_nsmallest_disjoint(data, n):
    """nlargest and nsmallest should be disjoint"""
    assume(len(set(data)) >= 2 * n)

    df = pd.DataFrame({'x': data})
    ddf = dd.from_pandas(df, npartitions=2)

    largest = set(ddf.nlargest(n, 'x')['x'].compute())
    smallest = set(ddf.nsmallest(n, 'x')['x'].compute())

    assert len(largest & smallest) == 0
```

**Failing input**: `data=[0, 0, 0, 0, 1], n=1`

## Reproducing the Bug

```python
import pandas as pd
import dask.dataframe as dd

df = pd.DataFrame({'x': [0, 0, 0, 0, 1]})
ddf = dd.from_pandas(df, npartitions=2)

result = ddf.nlargest(1, 'x')['x']
result.compute()
```

Output:
```
TypeError: Series.nlargest() got an unexpected keyword argument 'columns'
```

Full traceback shows the error originates in:
- `/dask/dataframe/dask_expr/_reductions.py:1339` in `NLargest.chunk()`
- The method calls `cls.reduction_chunk(df, **kwargs)` with `kwargs={'columns': 'x', 'n': 1}`
- This eventually calls `Series.nlargest(columns='x', n=1)`, but pandas Series.nlargest() only accepts `(n, keep='first')`

## Why This Is A Bug

This violates the expected behavior where column selection from a DataFrame operation should work seamlessly. The pattern `df.nlargest(n, col)[col]` is a common idiom that works in pandas but fails in dask. Users expect that selecting a single column from a DataFrame result should produce a valid Series operation.

The root cause is that NLargest stores `columns` as a parameter for DataFrame operations, but when the result is projected to a Series via `__getitem__`, it should not pass the `columns` parameter to the underlying pandas Series method.

## Fix

The fix should be in the `NLargest` class (and likely `NSmallest` as well) to detect when the result is a Series and avoid passing the `columns` parameter in that case. A possible approach:

```diff
--- a/dask/dataframe/dask_expr/_reductions.py
+++ b/dask/dataframe/dask_expr/_reductions.py
@@ -1336,7 +1336,12 @@ class NLargest(SingleAggregation):

     @classmethod
     def chunk(cls, df, **kwargs):
-        return cls.reduction_chunk(df, **kwargs)
+        # Remove 'columns' kwarg if operating on a Series
+        chunk_kwargs = kwargs.copy()
+        if isinstance(df, pd.Series) and 'columns' in chunk_kwargs:
+            chunk_kwargs.pop('columns')
+            chunk_kwargs.setdefault('n', kwargs.get('n'))
+        return cls.reduction_chunk(df, **chunk_kwargs)

     @classmethod
     def reduction_combine(cls, df, **kwargs):
```

Similar changes would be needed for `NSmallest` and potentially other reduction operations that have DataFrame-specific parameters but can be applied to Series results.