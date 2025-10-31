# Bug Report: dask.dataframe.io.orc Columns List Mutation

**Target**: `dask.dataframe.io.orc.core._read_orc`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `_read_orc` function mutates the `columns` parameter list by appending the index column, violating the principle that function parameters should not be modified unless explicitly documented.

## Property-Based Test

```python
import tempfile
import os
import pandas as pd
import pyarrow as pa
import pyarrow.orc as orc
from dask.dataframe.io.orc.core import _read_orc
from dask.dataframe.io.orc.arrow import ArrowORCEngine
import fsspec


def test_columns_list_mutation():
    with tempfile.TemporaryDirectory() as tmpdir:
        file1 = os.path.join(tmpdir, "file1.orc")
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "idx": [10, 11, 12]})

        with open(file1, "wb") as f:
            orc.write_table(pa.Table.from_pandas(df), f)

        fs = fsspec.filesystem("file")
        schema = {"a": "int64", "b": "int64", "idx": "int64"}
        columns_original = ["a", "b"]
        columns_copy = columns_original.copy()

        _read_orc(
            parts=[(file1, None)],
            engine=ArrowORCEngine,
            fs=fs,
            schema=schema,
            index="idx",
            columns=columns_original,
        )

        assert columns_original == columns_copy, f"columns list was mutated: {columns_copy} -> {columns_original}"
```

**Failing input**: `columns=["a", "b"]`, `index="idx"`

## Reproducing the Bug

```python
import tempfile
import os
import pandas as pd
import pyarrow as pa
import pyarrow.orc as orc
from dask.dataframe.io.orc.core import _read_orc
from dask.dataframe.io.orc.arrow import ArrowORCEngine
import fsspec

tmpdir = tempfile.mkdtemp()
file1 = os.path.join(tmpdir, "file1.orc")
df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "idx": [10, 11, 12]})

with open(file1, "wb") as f:
    orc.write_table(pa.Table.from_pandas(df), f)

fs = fsspec.filesystem("file")
schema = {"a": "int64", "b": "int64", "idx": "int64"}
columns = ["a", "b"]

print(f"Columns before: {columns}")

_read_orc(
    parts=[(file1, None)],
    engine=ArrowORCEngine,
    fs=fs,
    schema=schema,
    index="idx",
    columns=columns,
)

print(f"Columns after: {columns}")
```

## Why This Is A Bug

The function modifies the caller's list object, which is unexpected behavior and violates the principle of least surprise. If a caller reuses the `columns` list after calling this function, they will get unexpected results. While this is an internal function (prefixed with `_`), it's still called from `read_orc` via `dd.from_map`, and mutating parameters can lead to subtle bugs in concurrent or repeated operations.

## Fix

```diff
--- a/dask/dataframe/io/orc/core.py
+++ b/dask/dataframe/io/orc/core.py
@@ -110,7 +110,8 @@ def read_orc(

 def _read_orc(parts, *, engine, fs, schema, index, columns=None):
     if index is not None and columns is not None:
-        columns.append(index)
+        columns = columns + [index]
     _df = engine.read_partition(
         fs,
         parts,
```