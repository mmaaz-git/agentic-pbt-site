# Bug Report: dask.dataframe.io.orc._read_orc Columns Parameter Mutation

**Target**: `dask.dataframe.io.orc.core._read_orc`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `_read_orc` function mutates the `columns` parameter by appending the `index` to it, violating the principle that function parameters should not be modified.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings

@given(
    index_name=st.text(min_size=1, max_size=10),
    column_names=st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=5, unique=True)
)
@settings(max_examples=100)
def test_read_orc_does_not_mutate_columns_parameter(index_name, column_names):
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "test.orc")
        data = {col: [1, 2, 3] for col in column_names}
        data[index_name] = [10, 11, 12]
        df = pd.DataFrame(data)
        table = pa.Table.from_pandas(df)
        with open(file_path, "wb") as f:
            orc.write_table(table, f)

        fs = fsspec.filesystem("file")
        schema = {col: "int64" for col in df.columns}
        columns_original = list(column_names)
        columns_copy = columns_original.copy()

        _read_orc(
            parts=[(file_path, None)],
            engine=ArrowORCEngine,
            fs=fs,
            schema=schema,
            index=index_name,
            columns=columns_original,
        )

        assert columns_original == columns_copy
```

**Failing input**: Any case where `index` is not None and `columns` is not None

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

with tempfile.TemporaryDirectory() as tmpdir:
    file_path = os.path.join(tmpdir, "test.orc")
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "idx": [10, 11, 12]})
    table = pa.Table.from_pandas(df)
    with open(file_path, "wb") as f:
        orc.write_table(table, f)

    fs = fsspec.filesystem("file")
    schema = {"a": "int64", "b": "int64", "idx": "int64"}
    columns_original = ["a", "b"]
    columns_before = columns_original.copy()

    _read_orc(
        parts=[(file_path, None)],
        engine=ArrowORCEngine,
        fs=fs,
        schema=schema,
        index="idx",
        columns=columns_original,
    )

    print(f"Before: {columns_before}")
    print(f"After:  {columns_original}")
```

## Why This Is A Bug

Functions should not mutate their input parameters unless explicitly documented. The `columns` parameter is a list passed by the caller, and modifying it creates unexpected side effects. This violates Python's principle of least surprise.

## Fix

```diff
--- a/core.py
+++ b/core.py
@@ -110,7 +110,7 @@ def read_orc(

 def _read_orc(parts, *, engine, fs, schema, index, columns=None):
     if index is not None and columns is not None:
-        columns.append(index)
+        columns = columns + [index]
     _df = engine.read_partition(
         fs,
         parts,
```