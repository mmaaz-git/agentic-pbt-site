# Bug Report: dask.dataframe.io.orc Schema Validation Bypass

**Target**: `dask.dataframe.io.orc.arrow.ArrowORCEngine.read_metadata`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When `split_stripes=False`, the `read_metadata` method fails to validate schema consistency across multiple ORC files, always checking only the first file's schema instead of each file individually.

## Property-Based Test

```python
import tempfile
import os
import pandas as pd
import pyarrow as pa
import pyarrow.orc as orc
from dask.dataframe.io.orc.arrow import ArrowORCEngine
import fsspec


def test_schema_mismatch_not_detected_when_split_stripes_false():
    with tempfile.TemporaryDirectory() as tmpdir:
        file1 = os.path.join(tmpdir, "file1.orc")
        file2 = os.path.join(tmpdir, "file2.orc")

        df1 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        table1 = pa.Table.from_pandas(df1)

        df2 = pd.DataFrame({"a": [1, 2, 3], "c": [7, 8, 9]})
        table2 = pa.Table.from_pandas(df2)

        with open(file1, "wb") as f:
            orc.write_table(table1, f)
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
        assert False, "Should have raised ValueError for incompatible schemas"
```

**Failing input**: Two ORC files with different schemas: `{"a", "b"}` and `{"a", "c"}`

## Reproducing the Bug

```python
import tempfile
import os
import pandas as pd
import pyarrow as pa
import pyarrow.orc as orc
from dask.dataframe.io.orc.arrow import ArrowORCEngine
import fsspec

tmpdir = tempfile.mkdtemp()
file1 = os.path.join(tmpdir, "file1.orc")
file2 = os.path.join(tmpdir, "file2.orc")

df1 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
df2 = pd.DataFrame({"a": [1, 2, 3], "c": [7, 8, 9]})

with open(file1, "wb") as f:
    orc.write_table(pa.Table.from_pandas(df1), f)
with open(file2, "wb") as f:
    orc.write_table(pa.Table.from_pandas(df2), f)

fs = fsspec.filesystem("file")
parts, schema, meta = ArrowORCEngine.read_metadata(
    fs=fs,
    paths=[file1, file2],
    columns=None,
    index=None,
    split_stripes=False,
    aggregate_files=False,
)

print(f"Schema detected: {schema}")
print("No error raised - schema mismatch was not detected!")
```

## Why This Is A Bug

When reading multiple ORC files without stripe splitting, the code should validate that all files have compatible schemas (as it does when `split_stripes=True`). However, at line 60 in `arrow.py`, the code opens `paths[0]` instead of the current `path` variable, causing it to only check the first file's schema regardless of how many files are being processed. This allows incompatible files to be processed together, which will fail later during actual data reading with confusing errors.

## Fix

```diff
--- a/dask/dataframe/io/orc/arrow.py
+++ b/dask/dataframe/io/orc/arrow.py
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