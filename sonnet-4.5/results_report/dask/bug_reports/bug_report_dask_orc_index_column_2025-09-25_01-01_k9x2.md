# Bug Report: dask.dataframe.io.orc Index Column Not Included in Read

**Target**: `dask.dataframe.io.orc.read_orc`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

When calling `read_orc()` with a `columns` parameter that does not include the specified `index` column, the function crashes with a KeyError instead of automatically including the index column in the columns to read from the file.

## Property-Based Test

```python
import tempfile
import shutil
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, strategies as st, settings
import dask.dataframe as dd


@given(
    n_rows=st.integers(min_value=20, max_value=100),
    seed=st.integers(min_value=0, max_value=1000),
)
@settings(max_examples=50)
def test_columns_mutation_when_index_not_in_columns(tmp_path_factory, n_rows, seed):
    """
    Property: Should be able to read ORC with index column not in columns list.
    The index column should be automatically included in the read.
    """
    tmp = str(tmp_path_factory.mktemp("orc_test"))

    np.random.seed(seed)
    df_pandas = pd.DataFrame({
        "a": np.random.randint(0, 100, size=n_rows, dtype=np.int32),
        "b": np.random.randint(0, 100, size=n_rows, dtype=np.int32),
        "c": np.random.randint(0, 100, size=n_rows, dtype=np.int32),
    })

    df_dask = dd.from_pandas(df_pandas, npartitions=2)
    df_dask.to_orc(tmp, write_index=False)

    columns_to_read = ["a", "b"]
    index_col = "c"

    df_read = dd.read_orc(tmp, columns=columns_to_read, index=index_col)
    result = df_read.compute()

    assert list(result.columns) == ["a", "b"]
    assert result.index.name == "c"
```

**Failing input**: Any ORC file with columns=['a', 'b'], index='c' where 'c' is not in ['a', 'b']

## Reproducing the Bug

```python
import tempfile
import shutil
import pandas as pd
import dask.dataframe as dd

tmp = tempfile.mkdtemp()

df_pandas = pd.DataFrame({
    "a": [1, 2, 3, 4, 5],
    "b": [10, 20, 30, 40, 50],
    "c": [100, 200, 300, 400, 500],
})

df_dask = dd.from_pandas(df_pandas, npartitions=2)
df_dask.to_orc(tmp, write_index=False)

df_read = dd.read_orc(tmp, columns=["a", "b"], index="c")
result = df_read.compute()

shutil.rmtree(tmp, ignore_errors=True)
```

**Error**:
```
KeyError: "An error occurred while calling the read_orc method registered to the pandas backend.
Original Message: 'c'"
```

## Why This Is A Bug

The `read_orc` function accepts an `index` parameter to specify which column should be used as the index. When users specify a subset of columns to read via the `columns` parameter, they should not need to include the index column in that list - the function should automatically ensure the index column is read from the file so it can be set as the index.

This is analogous to how `pd.read_csv(columns=['a', 'b'], index_col='c')` works in pandas - the index column is automatically included in the read even if not explicitly listed in `columns`.

The current behavior violates the principle of least surprise and makes the API difficult to use correctly.

## Fix

The bug is in `/dask/dataframe/io/orc/arrow.py` at lines 76-78. The code needs to ensure that index columns are included in the list of columns to read before calling `_meta_from_dtypes`:

```diff
--- a/dask/dataframe/io/orc/arrow.py
+++ b/dask/dataframe/io/orc/arrow.py
@@ -75,6 +75,11 @@ class ArrowORCEngine:

         columns = list(schema) if columns is None else columns
         index = [index] if isinstance(index, str) else index
+        # Ensure index columns are included in columns to read
+        if index:
+            for idx_col in index:
+                if idx_col not in columns:
+                    columns.append(idx_col)
         meta = _meta_from_dtypes(columns, schema, index, [])
         return parts, schema, meta