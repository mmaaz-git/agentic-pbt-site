# Bug Report: dask.dataframe.io.orc _read_orc Mutates columns Parameter

**Target**: `dask.dataframe.io.orc.core._read_orc`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `_read_orc` function in `dask/dataframe/io/orc/core.py` mutates the `columns` parameter by calling `columns.append(index)`, violating the principle that function parameters should not be mutated unless explicitly documented.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pandas as pd
from dask.dataframe.io.orc.core import _read_orc
from unittest.mock import MagicMock

@given(st.lists(st.text(min_size=1), min_size=1, max_size=5, unique=True))
def test_columns_not_mutated_by_read_orc(col_names):
    original_columns = list(col_names)
    columns_copy = list(col_names)

    mock_engine = MagicMock()
    mock_fs = MagicMock()
    mock_schema = {col: 'int64' for col in col_names}

    df_data = {col: [1, 2, 3] for col in col_names}
    mock_df = pd.DataFrame(df_data)
    mock_engine.read_partition.return_value = mock_df

    parts = [("test.orc", None)]
    index_col = col_names[0]

    _read_orc(
        parts,
        engine=mock_engine,
        fs=mock_fs,
        schema=mock_schema,
        index=index_col,
        columns=columns_copy,
    )

    assert columns_copy == original_columns
```

**Failing input**: Any column list with an index specified, e.g., `columns=['a', 'b', 'c']` with `index='a'`

## Reproducing the Bug

```python
import pandas as pd
from dask.dataframe.io.orc.core import _read_orc
from unittest.mock import MagicMock

col_names = ['a', 'b', 'c']
columns_list = ['a', 'b', 'c']

mock_engine = MagicMock()
mock_fs = MagicMock()
mock_schema = {col: 'int64' for col in col_names}
df_data = {col: [1, 2, 3] for col in col_names}
mock_df = pd.DataFrame(df_data)
mock_engine.read_partition.return_value = mock_df

parts = [("test.orc", None)]

print(f"Before: columns_list = {columns_list}")

_read_orc(
    parts,
    engine=mock_engine,
    fs=mock_fs,
    schema=mock_schema,
    index='a',
    columns=columns_list,
)

print(f"After: columns_list = {columns_list}")
```

## Why This Is A Bug

The function mutates the caller's `columns` list by appending the `index` value to it (line 113 in `core.py`). This violates the principle of least surprise - callers don't expect their input lists to be modified. This can cause:

1. Unexpected behavior when the same columns list is reused
2. Potential issues in concurrent scenarios
3. The index value appearing twice in the columns list on subsequent calls

## Fix

```diff
--- a/dask/dataframe/io/orc/core.py
+++ b/dask/dataframe/io/orc/core.py
@@ -110,7 +110,8 @@ def from_map(

 def _read_orc(parts, *, engine, fs, schema, index, columns=None):
     if index is not None and columns is not None:
-        columns.append(index)
+        columns = columns + [index] if isinstance(index, str) else columns + index
+
     _df = engine.read_partition(
         fs,
         parts,
```