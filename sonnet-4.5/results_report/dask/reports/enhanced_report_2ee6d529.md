# Bug Report: dask.dataframe.io.orc.arrow.ArrowORCEngine Schema Validation Bypass When split_stripes=False

**Target**: `dask.dataframe.io.orc.arrow.ArrowORCEngine.read_metadata`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `read_metadata` method fails to validate schema consistency across multiple ORC files when `split_stripes=False`, incorrectly opening `paths[0]` instead of the current `path` in the iteration loop, causing it to only check the first file's schema repeatedly.

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

<details>

<summary>
**Failing input**: Two ORC files with schemas `{"a": int64, "b": int64}` and `{"a": int64, "c": int64}`
</summary>
```
Test FAILED - Should have raised ValueError for incompatible schemas
```
</details>

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

<details>

<summary>
Schema validation bypass - incompatible schemas not detected
</summary>
```
Schema detected: {'a': <class 'numpy.int64'>, 'b': <class 'numpy.int64'>}
No error raised - schema mismatch was not detected!
```
</details>

## Why This Is A Bug

When reading multiple ORC files, Dask should validate that all files have compatible schemas to prevent downstream errors during data reading. The code correctly performs this validation when `split_stripes=True` (lines 41-44 in arrow.py), but fails to do so when `split_stripes=False`.

The bug occurs at line 60 in `/dask/dataframe/io/orc/arrow.py`. In the else branch handling `split_stripes=False`, the code iterates through all paths but incorrectly opens `paths[0]` instead of the current `path` variable:

```python
for path in paths:
    if schema is None:
        with fs.open(paths[0], "rb") as f:  # BUG: Should be 'path', not 'paths[0]'
```

This causes the code to only ever check the first file's schema, regardless of how many files are being processed. Files with incompatible schemas pass through without raising the expected `ValueError("Incompatible schemas while parsing ORC files")`, leading to confusing errors later during actual data reading when the schema mismatch is encountered.

## Relevant Context

The ArrowORCEngine provides two modes for reading ORC files:
1. **With stripe splitting** (`split_stripes=True`): Divides files into smaller chunks based on ORC stripes and correctly validates schemas across all files
2. **Without stripe splitting** (`split_stripes=False`): Processes entire files as single units but fails to validate schemas due to this bug

The schema validation logic exists in the helper function `_get_schema` (lines 29-34) but is never called in the `split_stripes=False` branch. When `split_stripes=True`, the validation correctly raises an error for incompatible schemas.

Related code location: https://github.com/dask/dask/blob/main/dask/dataframe/io/orc/arrow.py

## Proposed Fix

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