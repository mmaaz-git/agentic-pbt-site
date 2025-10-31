# Bug Report: dask.dataframe.io.orc Index Not Preserved in Round-Trip

**Target**: `dask.dataframe.io.orc.read_orc` and `dask.dataframe.io.orc.to_orc`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When writing a Dask DataFrame to ORC with `write_index=True` and reading it back without specifying the `index` parameter, the index is not preserved. The index column is stored in the ORC file but is read as a regular column instead of being set as the index.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import tempfile
import shutil
import pandas as pd
import dask.dataframe as dd


@given(
    st.lists(st.integers(min_value=0, max_value=10), min_size=1, max_size=10),
    st.booleans(),
)
@settings(max_examples=100, deadline=None)
def test_orc_round_trip_preserves_index(data, write_index):
    tmpdir = tempfile.mkdtemp()
    try:
        df = pd.DataFrame({"a": data, "b": [x * 2 for x in data]})
        ddf = dd.from_pandas(df, npartitions=2)

        path = f"{tmpdir}/test_orc"
        ddf.to_orc(path, write_index=write_index)

        result = dd.read_orc(path)
        result_df = result.compute()

        if write_index:
            pd.testing.assert_frame_equal(result_df, df, check_dtype=False)
        else:
            pd.testing.assert_frame_equal(
                result_df.reset_index(drop=True), df, check_dtype=False
            )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
```

**Failing input**: `data=[0, 0], write_index=True`

## Reproducing the Bug

```python
import tempfile
import shutil
import pandas as pd
import dask.dataframe as dd

tmpdir = tempfile.mkdtemp()
try:
    df = pd.DataFrame({"a": [0, 0], "b": [0, 0]})
    print("Original DataFrame:")
    print(df)
    print(f"Original index: {df.index.tolist()}")

    ddf = dd.from_pandas(df, npartitions=2)
    path = f"{tmpdir}/test_orc"
    ddf.to_orc(path, write_index=True)

    result = dd.read_orc(path)
    result_df = result.compute()

    print("\nResult DataFrame:")
    print(result_df)
    print(f"Result index: {result_df.index.tolist()}")

    print("\nExpected index: [0, 1]")
    print(f"Actual index: {result_df.index.tolist()}")
    print("Index preserved:", result_df.index.tolist() == [0, 1])
finally:
    shutil.rmtree(tmpdir, ignore_errors=True)
```

Output:
```
Original DataFrame:
   a  b
0  0  0
1  0  0
Original index: [0, 1]

Result DataFrame:
   a  b  __index_level_0__
0  0  0                  0
0  0  0                  1
Result index: [0, 0]

Expected index: [0, 1]
Actual index: [0, 0]
Index preserved: False
```

## Why This Is A Bug

When `write_index=True` is specified, users expect the index to be preserved in a round-trip (write → read). The current implementation writes the index as a column named `__index_level_0__` but doesn't automatically restore it as the index when reading. This violates the fundamental round-trip property that `read_orc(to_orc(df, write_index=True))` should equal `df`.

The root cause is in `/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages/dask/dataframe/io/orc/arrow.py:106`:
```python
return pa.Table.from_batches(batches).to_pandas(date_as_object=False)
```

The PyArrow ORC format doesn't preserve pandas metadata about which columns are indices, so `to_pandas()` treats all columns as regular columns. The index information is lost during the ORC round-trip.

## Fix

The fix requires detecting and handling pandas index columns when reading ORC files. One approach is to check for columns with pandas index naming conventions (e.g., `__index_level_0__`) and automatically set them as indices:

```diff
--- a/dask/dataframe/io/orc/arrow.py
+++ b/dask/dataframe/io/orc/arrow.py
@@ -103,7 +103,15 @@ class ArrowORCEngine:
     def read_partition(cls, fs, parts, schema, columns, **kwargs):
         batches = []
         for path, stripes in parts:
             batches += _read_orc_stripes(fs, path, stripes, schema, columns)
-        return pa.Table.from_batches(batches).to_pandas(date_as_object=False)
+        table = pa.Table.from_batches(batches)
+        df = table.to_pandas(date_as_object=False)
+
+        # Restore pandas index columns
+        index_cols = [col for col in df.columns if col.startswith('__index_level_')]
+        if index_cols:
+            df = df.set_index(sorted(index_cols))
+            df.index.name = None if len(index_cols) == 1 and df.index.name == '__index_level_0__' else df.index.name
+
+        return df
```

Alternatively, document that `read_orc()` requires specifying the `index` parameter when reading files written with `write_index=True`, and consider changing the default to `write_index=False` to avoid this confusion.