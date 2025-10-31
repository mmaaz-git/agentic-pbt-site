# Bug Report: dask.dataframe.io.orc._read_orc Mutates Input Columns List

**Target**: `dask.dataframe.io.orc.core._read_orc`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The internal `_read_orc` function unexpectedly mutates the `columns` parameter by appending the index column to it, violating the principle that functions should not modify their input parameters without explicit documentation.

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


if __name__ == "__main__":
    test_columns_list_mutation()
    print("Test passed!")
```

<details>

<summary>
**Failing input**: `columns=["a", "b"], index="idx"`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/9/hypo.py", line 37, in <module>
    test_columns_list_mutation()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/9/hypo.py", line 33, in test_columns_list_mutation
    assert columns_original == columns_copy, f"columns list was mutated: {columns_copy} -> {columns_original}"
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: columns list was mutated: ['a', 'b'] -> ['a', 'b', 'idx']
```
</details>

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

<details>

<summary>
Columns list mutated after function call
</summary>
```
Columns before: ['a', 'b']
Columns after: ['a', 'b', 'idx']
```
</details>

## Why This Is A Bug

This violates the fundamental Python convention that functions should not mutate their input parameters unless explicitly documented to do so. The issue occurs in line 113 of `dask/dataframe/io/orc/core.py` where `columns.append(index)` directly modifies the caller's list object.

The mutation is particularly problematic because:
1. **No documentation** exists warning users that the `columns` parameter will be modified
2. **PEP 8 guidelines** suggest keeping functional behavior side-effect free
3. **The public API recognizes this issue**: The public `read_orc` function (lines 97-98) explicitly filters out the index from columns before calling `_read_orc`, suggesting the developers are aware this mutation occurs
4. **Potential for subtle bugs**: If code reuses the columns list after calling this function, unexpected behavior will occur
5. **Concurrent operations risk**: In multi-threaded scenarios or when the same columns list is used multiple times, this mutation could lead to race conditions or inconsistent state

While this is an internal function (prefixed with underscore), it still represents a contract violation that makes the codebase harder to maintain and reason about.

## Relevant Context

The `_read_orc` function is called internally by the public `read_orc` function via `dd.from_map` (line 99-108). The public function actually works around this mutation by pre-filtering the index from the columns list:

```python
# Line 97-98 in read_orc
if columns is not None and index in columns:
    columns = [col for col in columns if col != index]
```

This workaround indicates that the mutation behavior is recognized as problematic. The function needs the index column to be included in the columns list for the `engine.read_partition` call, but it should create a new list rather than mutating the input.

Relevant code location: `/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages/dask/dataframe/io/orc/core.py:111-123`

## Proposed Fix

```diff
--- a/dask/dataframe/io/orc/core.py
+++ b/dask/dataframe/io/orc/core.py
@@ -110,7 +110,7 @@ def read_orc(

 def _read_orc(parts, *, engine, fs, schema, index, columns=None):
     if index is not None and columns is not None:
-        columns.append(index)
+        columns = columns + [index]
     _df = engine.read_partition(
         fs,
         parts,
```