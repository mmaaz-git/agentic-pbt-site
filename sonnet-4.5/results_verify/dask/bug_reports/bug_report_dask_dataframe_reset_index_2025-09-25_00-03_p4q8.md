# Bug Report: dask.dataframe reset_index Produces Duplicate Indices

**Target**: `dask.dataframe.DataFrame.reset_index`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When calling `reset_index(drop=True)` on a Dask DataFrame with multiple partitions, each partition independently resets its index starting from 0, resulting in duplicate indices in the final computed result. This differs from pandas, which creates a sequential index across the entire DataFrame.

## Property-Based Test

```python
from hypothesis import given, settings
from hypothesis.extra.pandas import data_frames, column, range_indexes
import dask.dataframe as dd
import pandas as pd


@settings(max_examples=100)
@given(
    df=data_frames(
        columns=[
            column('x', dtype=int),
            column('y', dtype=int),
        ],
        index=range_indexes(min_size=1, max_size=50),
    ),
)
def test_reset_index_matches_pandas(df):
    ddf = dd.from_pandas(df, npartitions=2)

    dask_result = ddf.reset_index(drop=True).compute()
    pandas_result = df.reset_index(drop=True)

    pd.testing.assert_frame_equal(dask_result, pandas_result)
```

**Failing input**:
```python
df = pd.DataFrame({'x': [0, 0], 'y': [0, 0]})
```

## Reproducing the Bug

```python
import dask.dataframe as dd
import pandas as pd

df = pd.DataFrame({'x': [0, 0], 'y': [0, 0]})

pandas_result = df.reset_index(drop=True)
print("Pandas result:")
print(pandas_result)
print(f"Index: {pandas_result.index.tolist()}")

ddf = dd.from_pandas(df, npartitions=2)
dask_result = ddf.reset_index(drop=True).compute()
print("\nDask result:")
print(dask_result)
print(f"Index: {dask_result.index.tolist()}")
```

**Output:**
```
Pandas result:
   x  y
0  0  0
1  0  0
Index: [0, 1]

Dask result:
   x  y
0  0  0
0  0  0
Index: [0, 0]
```

## Why This Is A Bug

The `reset_index(drop=True)` operation should produce a sequential integer index starting from 0, just like pandas does. Instead, when a Dask DataFrame has multiple partitions, each partition resets its own index to start from 0, creating duplicate indices (both rows have index 0 in this example).

This violates the fundamental expectation that:
1. Dask operations should match pandas behavior
2. `reset_index(drop=True)` should create a unique, sequential index
3. The operation is documented to "reset the index, or a level of it"

The bug occurs whenever:
- `reset_index(drop=True)` is called on a Dask DataFrame
- The DataFrame has more than one partition
- Multiple rows exist across partitions

## Fix

The `reset_index` operation needs to coordinate across partitions to ensure sequential indices. Each partition after the first should offset its index by the cumulative count of rows in previous partitions. For example:
- Partition 0: rows get indices [0, 1, ...]
- Partition 1: rows get indices [len(partition_0), len(partition_0) + 1, ...]
- And so on

The fix should be in the implementation of `reset_index` to properly handle multi-partition DataFrames by maintaining index continuity across partition boundaries.