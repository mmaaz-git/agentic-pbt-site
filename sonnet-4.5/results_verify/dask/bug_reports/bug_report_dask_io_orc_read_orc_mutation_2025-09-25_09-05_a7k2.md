# Bug Report: dask.dataframe.io.orc._read_orc Mutates Input Columns List

**Target**: `dask.dataframe.io.orc.core._read_orc`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_read_orc` function mutates the input `columns` list by appending the `index` column to it, violating the principle that functions should not modify their arguments.

## Property-Based Test

```python
from unittest.mock import Mock, MagicMock
from hypothesis import given, strategies as st, settings
from dask.dataframe.io.orc.core import _read_orc

@given(st.lists(st.text(min_size=1), min_size=1), st.text(min_size=1))
@settings(max_examples=200)
def test_read_orc_does_not_mutate_columns(columns_list, index_name):
    original_columns = columns_list.copy()

    mock_engine = Mock()
    mock_engine.read_partition = Mock(return_value=Mock())

    mock_df = MagicMock()
    mock_df.set_index = Mock(return_value=MagicMock())
    mock_engine.read_partition.return_value = mock_df

    try:
        _read_orc(
            parts=[],
            engine=mock_engine,
            fs=Mock(),
            schema={},
            index=index_name,
            columns=columns_list
        )
    except Exception:
        pass

    assert columns_list == original_columns, "columns list should not be mutated"
```

**Failing input**: `columns_list=['0'], index_name='0'`

## Reproducing the Bug

```python
from unittest.mock import Mock, MagicMock
from dask.dataframe.io.orc.core import _read_orc

columns_list = ['col1', 'col2']
index_name = 'col1'

print("Before:", columns_list)

mock_engine = Mock()
mock_df = MagicMock()
mock_df.set_index = Mock(return_value=MagicMock())
mock_engine.read_partition.return_value = mock_df

_read_orc(
    parts=[],
    engine=mock_engine,
    fs=Mock(),
    schema={},
    index=index_name,
    columns=columns_list
)

print("After:", columns_list)
```

## Why This Is A Bug

Functions should not mutate their input arguments. This violates standard Python conventions and can cause unexpected behavior if the caller reuses the `columns` list. The mutation happens at line 113 in `core.py` where `columns.append(index)` directly modifies the input list.

## Fix

```diff
--- a/dask/dataframe/io/orc/core.py
+++ b/dask/dataframe/io/orc/core.py
@@ -110,7 +110,7 @@ def _read_orc(parts, *, engine, fs, schema, index, columns=None):

 def _read_orc(parts, *, engine, fs, schema, index, columns=None):
     if index is not None and columns is not None:
-        columns.append(index)
+        columns = columns + [index]
     _df = engine.read_partition(
         fs,
         parts,
```