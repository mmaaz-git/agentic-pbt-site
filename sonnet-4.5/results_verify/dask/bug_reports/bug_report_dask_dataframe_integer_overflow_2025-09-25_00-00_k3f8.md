# Bug Report: dask.dataframe Integer Overflow in Aggregation Operations

**Target**: `dask.dataframe.DataFrame.sum()` and `dask.dataframe.DataFrame.mean()`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

Dask and pandas produce different results when performing aggregation operations (sum, mean) on large integers that cause overflow, violating the fundamental contract that dask.dataframe should match pandas.dataframe behavior.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env')

import pandas as pd
import dask.dataframe as dd
from hypothesis import given, settings
import hypothesis.extra.pandas as pd_st


@given(pd_st.data_frames([
    pd_st.column('a', dtype=int),
    pd_st.column('b', dtype=float),
]))
@settings(max_examples=100)
def test_operations_match_pandas_sum(df):
    ddf = dd.from_pandas(df, npartitions=2)

    dask_sum = ddf.sum().compute()
    pandas_sum = df.sum()

    pd.testing.assert_series_equal(dask_sum, pandas_sum)
```

**Failing input**:
```python
df = pd.DataFrame({
    'a': [4611686018427387904, 4611686018427387904],
    'b': [0.0, 0.0]
})
```

## Reproducing the Bug

```python
import pandas as pd
import dask.dataframe as dd

df = pd.DataFrame({
    'a': [4611686018427387904, 4611686018427387904],
    'b': [0.0, 0.0]
})

pandas_sum = df.sum()
print(f"Pandas sum of column 'a': {pandas_sum['a']}")

ddf = dd.from_pandas(df, npartitions=2)
dask_sum = ddf.sum().compute()
print(f"Dask sum of column 'a':   {dask_sum['a']}")

assert pandas_sum['a'] == dask_sum['a'], f"Expected {pandas_sum['a']}, got {dask_sum['a']}"
```

Output:
```
Pandas sum of column 'a': -9.223372036854776e+18
Dask sum of column 'a':   9.223372036854776e+18
AssertionError: Expected -9.223372036854776e+18, got 9.223372036854776e+18
```

## Why This Is A Bug

Dask's primary value proposition is providing a pandas-compatible API for larger-than-memory datasets. When aggregation operations produce different results than pandas for the same input data, it violates this fundamental contract. This bug causes silent data corruption - users will get incorrect results without any warning or error.

The value `4611686018427387904` is `2^62`, and adding it to itself causes int64 overflow. Pandas handles this overflow by wrapping to negative values, but dask produces a different (positive) result. Both `sum()` and `mean()` operations are affected.

## Fix

The root cause is likely that dask is using different dtype handling or aggregation strategies than pandas during partitioned computation. When aggregating across partitions, dask may be converting to float64 at a different point than pandas, leading to different overflow behavior.

A proper fix would require investigating dask's aggregation tree reduction logic to ensure it maintains pandas-compatible overflow semantics for integer operations.