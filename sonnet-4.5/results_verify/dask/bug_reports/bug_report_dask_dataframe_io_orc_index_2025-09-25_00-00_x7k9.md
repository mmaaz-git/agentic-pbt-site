# Bug Report: dask.dataframe.io.orc Index Duplication

**Target**: `dask.dataframe.io.orc.to_orc` and `dask.dataframe.io.orc.read_orc`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When writing a Dask DataFrame with `write_index=False` and reading it back, the resulting DataFrame has duplicate index values instead of a continuous RangeIndex. Each partition's index is reset independently, causing index values to repeat across partitions.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env')

import tempfile
import os
import dask.dataframe as dd
import pandas as pd
from hypothesis import given, settings
from hypothesis.extra.pandas import column, data_frames, range_indexes


@given(
    df=data_frames(
        columns=[
            column("a", dtype=int),
            column("b", dtype=float),
        ],
        index=range_indexes(min_size=1, max_size=100),
    )
)
@settings(max_examples=50, deadline=None)
def test_orc_write_index_false_no_duplicates(df):
    """When write_index=False, reading back should have continuous index"""
    ddf = dd.from_pandas(df, npartitions=2)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_data")
        ddf.to_orc(path, write_index=False)
        result = dd.read_orc(path).compute()

        assert not result.index.has_duplicates, f"Index has duplicates: {result.index.tolist()}"
```

**Failing input**: Any DataFrame with 2 or more rows when split into multiple partitions

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env')

import tempfile
import os
import dask.dataframe as dd
import pandas as pd

df = pd.DataFrame({'a': [0, 1, 2, 3], 'b': [0.0, 1.0, 2.0, 3.0]})
ddf = dd.from_pandas(df, npartitions=2)

with tempfile.TemporaryDirectory() as tmpdir:
    path = os.path.join(tmpdir, "test_data")
    ddf.to_orc(path, write_index=False)
    result = dd.read_orc(path).compute()

    print("Expected: RangeIndex(0, 1, 2, 3)")
    print(f"Actual:   {result.index.tolist()}")
    assert result.index.tolist() == [0, 1, 2, 3], f"Got {result.index.tolist()}"
```

## Why This Is A Bug

When `write_index=False` is specified, users expect the index to be dropped. After reading the data back, the index should be a continuous RangeIndex starting from 0. Instead, each partition gets its own index reset to [0, 1, ...], and these are concatenated, creating duplicate index values.

This violates the fundamental DataFrame invariant that indexes should uniquely identify rows (unless explicitly creating MultiIndex or duplicate indexes). Users who rely on unique index values will encounter unexpected behavior.

## Fix

The issue is that when reading ORC files, each partition is read independently with its own index starting from 0. The fix should ensure that when partitions are concatenated, the index is either:
1. Reset to a continuous RangeIndex across all partitions, or
2. Use `ignore_index=True` when concatenating partitions

A potential fix location would be in the `read_orc` function or in the partition reading logic to track and adjust index values across partitions, similar to how `pd.concat([...], ignore_index=True)` works.

The core reading happens via `dd.from_map(_read_orc, parts, ...)` in the `read_orc` function. The fix could involve either:
- Adding index offset information to each partition read
- Post-processing the result to reset the index to a continuous range