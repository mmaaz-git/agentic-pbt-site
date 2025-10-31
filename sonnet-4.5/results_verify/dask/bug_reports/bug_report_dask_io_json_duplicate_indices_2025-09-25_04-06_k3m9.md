# Bug Report: dask.dataframe.io read_json Creates Duplicate Indices

**Target**: `dask.dataframe.io.read_json`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When reading JSON files written by `to_json` with multiple partitions, `read_json` creates duplicate indices. Each partition's index starts from 0, causing index values to be duplicated across partitions instead of being sequential.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume, settings
from hypothesis.extra.pandas import data_frames, columns
import tempfile
import shutil
import os
import pandas as pd
import dask.dataframe as dd


@given(
    df=data_frames(
        columns=columns(['A', 'B', 'C'], dtype=float),
        rows=st.tuples(st.just(1), st.integers(min_value=1, max_value=20))
    ),
    npartitions=st.integers(min_value=2, max_value=3)
)
@settings(max_examples=100, deadline=None)
def test_json_round_trip_preserves_order(df, npartitions):
    """JSON round-trip should not create duplicate indices"""
    assume(len(df) >= npartitions)

    tmpdir = tempfile.mkdtemp()
    try:
        ddf = dd.from_pandas(df, npartitions=npartitions)
        path = os.path.join(tmpdir, 'data-*.json')
        dd.io.to_json(ddf, path, orient='records', lines=True)

        result_ddf = dd.io.read_json(path, orient='records', lines=True)
        result = result_ddf.compute()

        assert not result.index.duplicated().any(), \
            f"Index has duplicates: {list(result.index)}"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
```

**Failing input**: Any DataFrame with npartitions >= 2

## Reproducing the Bug

```python
import tempfile
import shutil
import os
import pandas as pd
import dask.dataframe as dd

df = pd.DataFrame({'x': [10, 20, 30]})
print(f"Original index: {list(df.index)}")

tmpdir = tempfile.mkdtemp()
try:
    ddf = dd.from_pandas(df, npartitions=2)
    path = os.path.join(tmpdir, 'data-*.json')
    dd.io.to_json(ddf, path, orient='records', lines=True)

    result_ddf = dd.io.read_json(path, orient='records', lines=True)
    result = result_ddf.compute()

    print(f"Result index: {list(result.index)}")
    print(f"Has duplicates: {result.index.duplicated().any()}")
finally:
    shutil.rmtree(tmpdir, ignore_errors=True)
```

**Output:**
```
Original index: [0, 1, 2]
Result index: [0, 1, 0]
Has duplicates: True
```

## Why This Is A Bug

The round-trip property is violated: `read_json(to_json(df))` should preserve the DataFrame's structure, including having a valid, sequential index. Instead, when reading multiple JSON files (one per partition), each file gets its own index starting from 0, creating duplicate indices.

This violates the expected behavior that:
1. A round-trip through JSON should preserve data
2. Reading multiple JSON files should behave like reading a single concatenated file
3. Pandas' `read_json` creates sequential indices when reading a single file

## Fix

The issue is in how `read_json` assigns indices when reading multiple files. Each partition should have its index offset by the cumulative size of previous partitions. A potential fix would involve:

1. Computing the cumulative sizes of partitions
2. Assigning each partition an index range like `[start, start+length)`

This is similar to how pandas handles reading a single JSON file - it assigns sequential indices `[0, 1, 2, ...]` rather than all rows getting index 0.