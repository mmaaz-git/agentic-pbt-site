# Bug Report: dask.dataframe.io.orc.read_orc Index Column Not In Columns List

**Target**: `dask.dataframe.io.orc.read_orc`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

When calling `read_orc()` with an `index` parameter that specifies a column not included in the `columns` parameter, the function crashes with a `KeyError` instead of correctly reading the index column.

## Property-Based Test

```python
import tempfile
import pandas as pd
import dask.dataframe as dd
from hypothesis import given, strategies as st, settings


@settings(max_examples=50)
@given(
    nrows=st.integers(min_value=20, max_value=100),
    npartitions=st.integers(min_value=2, max_value=5),
)
def test_read_orc_columns_not_duplicated_across_partitions(nrows, npartitions):
    with tempfile.TemporaryDirectory() as tmpdir:
        data = pd.DataFrame({
            'a': range(nrows),
            'b': range(nrows, 2 * nrows),
            'c': range(2 * nrows, 3 * nrows),
        })
        df = dd.from_pandas(data, npartitions=npartitions)
        df.to_orc(tmpdir, write_index=False)

        result = dd.read_orc(tmpdir, columns=['b', 'c'], index='a')
        result_df = result.compute()

        assert 'a' not in result_df.columns
        assert list(result_df.columns) == ['b', 'c']
        assert result_df.index.name == 'a'
```

**Failing input**: Any valid ORC file with columns `['a', 'b', 'c']` when read with `columns=['b', 'c']` and `index='a'`

## Reproducing the Bug

```python
import tempfile
import pandas as pd
import dask.dataframe as dd

with tempfile.TemporaryDirectory() as tmpdir:
    data = pd.DataFrame({
        'a': [1, 2, 3],
        'b': [4, 5, 6],
        'c': [7, 8, 9],
    })
    df = dd.from_pandas(data, npartitions=1)
    df.to_orc(tmpdir, write_index=False)

    result = dd.read_orc(tmpdir, columns=['b', 'c'], index='a')
    result.compute()
```

This raises:
```
KeyError: "An error occurred while calling the read_orc method registered to the pandas backend.
Original Message: 'a'"
```

## Why This Is A Bug

The `read_orc` function accepts both a `columns` parameter (to select which columns to read) and an `index` parameter (to specify which column to use as the index). It's reasonable for users to specify an index column that isn't in the `columns` list, as the index column should be read automatically.

The bug occurs because:

1. In `read_orc()` at core.py:97-98, the code only handles the case where `index in columns`:
   ```python
   if columns is not None and index in columns:
       columns = [col for col in columns if col != index]
   ```

2. When `index` is NOT in `columns`, this check fails and `columns` is passed unchanged to `engine.read_metadata()` at line 88-95.

3. Later in `arrow.py:78`, `_meta_from_dtypes` is called with `columns=['b', 'c']` and `index=['a']`.

4. In `io/utils.py:117`, it tries to pop the index column from the data dict:
   ```python
   indexes = [data.pop(c) for c in index_cols or []]
   ```

5. But 'a' was never added to the data dict because it wasn't in the columns list, causing a `KeyError`.

## Fix

The fix requires ensuring that when an index is specified, it's included in the columns passed to `engine.read_metadata()` even if not explicitly requested by the user.

```diff
diff --git a/dask/dataframe/io/orc/core.py b/dask/dataframe/io/orc/core.py
index 1234567..abcdefg 100644
--- a/dask/dataframe/io/orc/core.py
+++ b/dask/dataframe/io/orc/core.py
@@ -82,6 +82,11 @@ def read_orc(
         path, mode="rb", storage_options=storage_options
     )

+    # Ensure index column is included when reading metadata
+    columns_for_read = columns
+    if columns is not None and index is not None and index not in columns:
+        columns_for_read = columns + [index]
+
     # Let backend engine generate a list of parts
     # from the ORC metadata.  The backend should also
     # return the schema and DataFrame-collection metadata
     parts, schema, meta = engine.read_metadata(
         fs,
         paths,
-        columns,
+        columns_for_read,
         index,
         split_stripes,
         aggregate_files,
```