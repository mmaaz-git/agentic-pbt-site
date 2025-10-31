# Bug Report: dask.dataframe.reset_index Creates Duplicate Index Values

**Target**: `dask.dataframe.DataFrame.reset_index`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When calling `reset_index()` on a dask DataFrame split across multiple partitions, the resulting index contains duplicate values (all 0s) instead of a sequential RangeIndex like pandas produces.

## Property-Based Test

```python
import numpy as np
import pandas as pd
from hypothesis import given, strategies as st, settings
import dask.dataframe as dd


@given(st.integers(min_value=1, max_value=30))
@settings(max_examples=300)
def test_reset_index_round_trip(n):
    df = pd.DataFrame({
        'a': np.random.randint(0, 100, n)
    })
    df.index = pd.Index(range(10, 10+n), name='idx')

    ddf = dd.from_pandas(df, npartitions=2)

    reset = ddf.reset_index()
    result = reset.compute()

    expected = df.reset_index()
    pd.testing.assert_frame_equal(result, expected)
```

**Failing input**: `n=2`

## Reproducing the Bug

```python
import pandas as pd
import dask.dataframe as dd

df = pd.DataFrame({'a': [1, 2]})
df.index = pd.Index([10, 11], name='idx')

ddf = dd.from_pandas(df, npartitions=2)
result = ddf.reset_index().compute()

print(result.index.tolist())
```

**Output**: `[0, 0]` (incorrect)
**Expected**: `[0, 1]`

## Why This Is A Bug

According to pandas documentation and behavior, `reset_index()` should create a new RangeIndex starting from 0 with sequential integers. Dask's implementation produces duplicate index values when the DataFrame is split across multiple partitions, with each partition starting its index at 0 instead of continuing from the previous partition's end.

## Fix

The issue is that each partition is resetting its index independently, starting from 0. The fix should ensure that when partitions are concatenated, the index continues sequentially. This likely requires passing offset information to each partition during the reset_index operation, or reconstructing the index after concatenation.