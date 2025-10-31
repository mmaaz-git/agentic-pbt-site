# Bug Report: dask.dataframe.io.orc.read_orc Index Column Not in Columns List

**Target**: `dask.dataframe.io.orc.read_orc`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

When calling `read_orc()` with a `columns` parameter that does not include the column specified as `index`, the function crashes with a KeyError instead of automatically including the index column in the read operation.

## Property-Based Test

```python
import tempfile
import shutil
import pandas as pd
from hypothesis import given, strategies as st, settings
import dask.dataframe as dd


@given(
    st.lists(st.integers(min_value=0, max_value=100), min_size=1, max_size=20),
)
@settings(max_examples=100, deadline=10000)
def test_index_not_in_columns_should_work(data):
    tmpdir = tempfile.mkdtemp()
    try:
        df = pd.DataFrame({"a": data, "b": data, "c": data})
        ddf = dd.from_pandas(df, npartitions=2)

        orc_path = f"{tmpdir}/test.orc"
        ddf.to_orc(orc_path)

        result = dd.read_orc(orc_path, columns=["a"], index="c")
        computed = result.compute()

        assert len(computed) == len(df)
        assert list(computed.columns) == ["a"]
        assert computed.index.name == "c"

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
```

**Failing input**: Any valid DataFrame

## Reproducing the Bug

```python
import tempfile
import shutil
import pandas as pd
import dask.dataframe as dd

tmpdir = tempfile.mkdtemp()
try:
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    ddf = dd.from_pandas(df, npartitions=1)

    orc_path = f"{tmpdir}/test.orc"
    ddf.to_orc(orc_path)

    result = dd.read_orc(orc_path, columns=["a"], index="c")
    print(result.compute())

finally:
    shutil.rmtree(tmpdir, ignore_errors=True)
```

**Error**:
```
KeyError: "An error occurred while calling the read_orc method registered to the pandas backend.
Original Message: 'c'"
```

## Why This Is A Bug

The `read_orc` function documentation states:
- `columns`: None or list(str) - Columns to load. If None, loads all.
- `index`: str - Column name to set as index.

There is no requirement that the index must be included in the columns list. The function should automatically include the index column when reading, similar to how `columns=["a", "c"], index="c"` works correctly and returns a DataFrame with column "a" and index "c".

The current behavior is inconsistent:
- `read_orc(path, columns=["a", "c"], index="c")` ✓ Works
- `read_orc(path, columns=["a"], index="c")` ✗ Crashes with KeyError

## Fix

```diff
--- a/dask/dataframe/io/orc/arrow.py
+++ b/dask/dataframe/io/orc/arrow.py
@@ -73,8 +73,14 @@ class ArrowORCEngine:
         # Check if we can aggregate adjacent parts together
         parts = cls._aggregate_files(aggregate_files, split_stripes, parts)

         columns = list(schema) if columns is None else columns
         index = [index] if isinstance(index, str) else index
+        # Ensure index columns are included when building metadata
+        if index:
+            columns_for_meta = list(columns)
+            for idx in index:
+                if idx not in columns_for_meta:
+                    columns_for_meta.append(idx)
+        else:
+            columns_for_meta = columns
-        meta = _meta_from_dtypes(columns, schema, index, [])
+        meta = _meta_from_dtypes(columns_for_meta, schema, index, [])
         return parts, schema, meta
```