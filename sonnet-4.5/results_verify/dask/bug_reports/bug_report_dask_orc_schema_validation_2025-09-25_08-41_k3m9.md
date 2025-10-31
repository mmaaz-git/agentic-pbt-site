# Bug Report: dask.dataframe.io.orc Schema Validation Skipped for Multiple Files

**Target**: `dask.dataframe.io.orc.arrow.ArrowORCEngine.read_metadata`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When `split_stripes=False`, the schema validation loop opens `paths[0]` instead of `path`, causing it to only validate the first file's schema and silently ignore schema mismatches in subsequent files.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import pytest

@given(num_files=st.integers(min_value=2, max_value=5))
@settings(max_examples=50)
def test_read_metadata_detects_schema_mismatch_with_split_stripes_false(num_files):
    with tempfile.TemporaryDirectory() as tmpdir:
        files = []
        for i in range(num_files):
            file_path = os.path.join(tmpdir, f"file{i}.orc")
            if i == num_files - 1:
                df = pd.DataFrame({"different_col": [1, 2, 3]})
            else:
                df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
            table = pa.Table.from_pandas(df)
            with open(file_path, "wb") as f:
                orc.write_table(table, f)
            files.append(file_path)

        fs = fsspec.filesystem("file")

        with pytest.raises(ValueError, match="Incompatible schemas"):
            ArrowORCEngine.read_metadata(
                fs=fs,
                paths=files,
                columns=None,
                index=None,
                split_stripes=False,
                aggregate_files=False,
            )
```

**Failing input**: Any case where multiple ORC files have different schemas and `split_stripes=False`

## Reproducing the Bug

```python
import tempfile
import os
import pandas as pd
import pyarrow as pa
import pyarrow.orc as orc
from dask.dataframe.io.orc.arrow import ArrowORCEngine
import fsspec

with tempfile.TemporaryDirectory() as tmpdir:
    file1 = os.path.join(tmpdir, "file1.orc")
    file2 = os.path.join(tmpdir, "file2.orc")

    df1 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    table1 = pa.Table.from_pandas(df1)
    with open(file1, "wb") as f:
        orc.write_table(table1, f)

    df2 = pd.DataFrame({"a": [1, 2, 3], "c": [7, 8, 9]})
    table2 = pa.Table.from_pandas(df2)
    with open(file2, "wb") as f:
        orc.write_table(table2, f)

    fs = fsspec.filesystem("file")

    parts, schema, meta = ArrowORCEngine.read_metadata(
        fs=fs,
        paths=[file1, file2],
        columns=None,
        index=None,
        split_stripes=False,
        aggregate_files=False,
    )

    print(f"No error raised - schema mismatch not detected!")
    print(f"Schema: {schema}")
```

## Why This Is A Bug

When processing multiple ORC files with `split_stripes=False`, the code should validate that all files have compatible schemas. However, line 60 in `arrow.py` opens `paths[0]` instead of the loop variable `path`, so it only checks the first file's schema repeatedly, missing mismatches in other files. This can lead to runtime errors or data corruption when reading incompatible files.

## Fix

```diff
--- a/arrow.py
+++ b/arrow.py
@@ -57,7 +57,7 @@ class ArrowORCEngine:
         else:
             for path in paths:
                 if schema is None:
-                    with fs.open(paths[0], "rb") as f:
+                    with fs.open(path, "rb") as f:
                         o = orc.ORCFile(f)
                         schema = o.schema
                 parts.append([(path, None)])
```