# Bug Report: dask.dataframe.io.orc Empty DataFrame Crash

**Target**: `dask.dataframe.io.orc.read_orc`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `read_orc` function crashes with `ValueError: All iterables must have a non-zero length` when reading ORC files written from empty DataFrames, despite empty DataFrames being valid and `to_orc` successfully writing them.

## Property-Based Test

```python
import tempfile
import shutil
import pandas as pd
import dask.dataframe as dd
from hypothesis import given, strategies as st, settings


@settings(max_examples=50, deadline=10000)
@given(
    num_rows=st.integers(min_value=0, max_value=50),
)
def test_orc_empty_dataframe_round_trip(num_rows):
    df_pandas = pd.DataFrame({
        'col_a': list(range(num_rows)),
        'col_b': [f'val_{i}' for i in range(num_rows)],
    })
    df = dd.from_pandas(df_pandas, npartitions=1)

    tmpdir = tempfile.mkdtemp()
    try:
        dd.to_orc(df, tmpdir, write_index=False)
        df_read = dd.read_orc(tmpdir)
        df_result = df_read.compute()

        assert len(df_result) == num_rows, \
            f"Row count mismatch: {len(df_result)} != {num_rows}"

        if num_rows == 0:
            assert list(df_result.columns) == ['col_a', 'col_b'], \
                "Column names should be preserved even for empty dataframes"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
```

**Failing input**: `num_rows=0`

## Reproducing the Bug

```python
import tempfile
import shutil
import pandas as pd
import dask.dataframe as dd

tmpdir = tempfile.mkdtemp()
try:
    df_pandas = pd.DataFrame({'col_a': [], 'col_b': []})
    df = dd.from_pandas(df_pandas, npartitions=1)

    dd.to_orc(df, tmpdir, write_index=False)
    df_read = dd.read_orc(tmpdir)

except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
finally:
    shutil.rmtree(tmpdir, ignore_errors=True)
```

## Why This Is A Bug

The round-trip property is fundamental to I/O operations: any data that can be written should be readable. Empty DataFrames are valid in both pandas and dask, and `to_orc` successfully writes them to disk. PyArrow can read these files without issue. However, `read_orc` crashes instead of returning the empty DataFrame, violating the round-trip invariant.

## Fix

The bug occurs in `read_orc` when calling `dd.from_map()` with an empty `parts` list. The fix should check for empty parts and handle that case appropriately:

```diff
--- a/dask/dataframe/io/orc/core.py
+++ b/dask/dataframe/io/orc/core.py
@@ -96,6 +96,10 @@ def read_orc(

     if columns is not None and index in columns:
         columns = [col for col in columns if col != index]
+
+    if not parts:
+        # Handle empty ORC files
+        return dd.from_pandas(meta, npartitions=1)
+
     return dd.from_map(
         _read_orc,
         parts,
```