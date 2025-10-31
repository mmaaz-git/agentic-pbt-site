# Bug Report: dask.dataframe.io.orc._read_orc Mutates Columns Parameter

**Target**: `dask.dataframe.io.orc.core._read_orc`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_read_orc` function mutates the `columns` list parameter by appending the index column to it, violating the principle that functions should not have side effects on input parameters.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st
import pyarrow as pa
from dask.dataframe.io.orc.arrow import ArrowORCEngine
from dask.dataframe.io.orc.core import _read_orc


@given(
    st.lists(st.text(min_size=1, max_size=10, alphabet=st.characters(min_codepoint=97, max_codepoint=122)), min_size=1, max_size=5, unique=True),
    st.text(min_size=1, max_size=10, alphabet=st.characters(min_codepoint=97, max_codepoint=122))
)
@settings(max_examples=100)
def test_read_orc_does_not_mutate_columns(column_names, index_name):
    columns_list = list(column_names)
    original_columns = columns_list.copy()

    parts = [("dummy_path", [0])]
    engine = ArrowORCEngine
    fs = None
    schema = {col: pa.int64() for col in column_names}

    try:
        _read_orc(parts, engine=engine, fs=fs, schema=schema, index=index_name, columns=columns_list)
    except:
        pass

    assert columns_list == original_columns, f"columns list was mutated: {original_columns} -> {columns_list}"
```

**Failing input**: `column_names=['a'], index_name='a'`

## Reproducing the Bug

```python
from dask.dataframe.io.orc.core import _read_orc
from dask.dataframe.io.orc.arrow import ArrowORCEngine

columns_list = ['col1', 'col2']
original = columns_list.copy()

parts = [("dummy_path", [0])]
try:
    _read_orc(parts, engine=ArrowORCEngine, fs=None, schema={}, index='col1', columns=columns_list)
except:
    pass

print(f"Before: {original}")
print(f"After:  {columns_list}")
print(f"Mutated: {columns_list != original}")
```

## Why This Is A Bug

Functions should not mutate their input parameters as this creates unexpected side effects. If a caller passes a columns list and reuses it, they will get unexpected behavior with duplicate entries. This violates basic principles of functional programming and can lead to subtle bugs in calling code.

The issue occurs at `core.py:113` where `columns.append(index)` directly modifies the input list instead of creating a new list.

## Fix

```diff
--- a/dask/dataframe/io/orc/core.py
+++ b/dask/dataframe/io/orc/core.py
@@ -110,7 +110,10 @@ def to_orc(

 def _read_orc(parts, *, engine, fs, schema, index, columns=None):
     if index is not None and columns is not None:
-        columns.append(index)
+        columns = columns + [index]
+    elif columns is not None:
+        columns = list(columns)
+
     _df = engine.read_partition(
         fs,
         parts,
```