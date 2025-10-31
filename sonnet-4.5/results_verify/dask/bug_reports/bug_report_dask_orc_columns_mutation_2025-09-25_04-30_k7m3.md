# Bug Report: dask.dataframe.io.orc Columns Parameter Mutation

**Target**: `dask.dataframe.io.orc.core._read_orc`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_read_orc` function mutates its `columns` parameter by appending the index column to it, violating Python's convention that function parameters should not be modified unless explicitly documented.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env')

import tempfile
import os
import pandas as pd
import pyarrow as pa
import pyarrow.orc as orc
from hypothesis import given, strategies as st, settings, assume
from dask.dataframe.io.orc.arrow import ArrowORCEngine
from dask.dataframe.io.orc.core import _read_orc


@settings(max_examples=100)
@given(
    st.lists(st.integers(min_value=0, max_value=100), min_size=1, max_size=20),
    st.lists(st.text(min_size=1, max_size=10, alphabet=st.characters(min_codepoint=97, max_codepoint=122)), min_size=1, max_size=5).map(lambda x: list(set(x)))
)
def test_columns_not_mutated_in_read_orc(data, columns):
    assume(len(columns) >= 2)

    with tempfile.TemporaryDirectory() as tmpdir:
        df_data = {col: data for col in columns}
        df = pd.DataFrame(df_data)
        table = pa.Table.from_pandas(df)

        path = os.path.join(tmpdir, "test.orc")
        with open(path, 'wb') as f:
            orc.write_table(table, f)

        class FakeFS:
            def open(self, path, mode):
                return open(path, mode)

        fs = FakeFS()
        index_col = columns[0]
        read_columns = columns[1:].copy()
        original_columns = read_columns.copy()

        parts = [(path, None)]
        schema = {col: df[col].dtype for col in columns}

        _read_orc(
            parts,
            engine=ArrowORCEngine,
            fs=fs,
            schema=schema,
            index=index_col,
            columns=read_columns
        )

        assert read_columns == original_columns
```

**Failing input**: `data=[0], columns=['aa', 'a']`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env')

import tempfile
import os
import pandas as pd
import pyarrow as pa
import pyarrow.orc as orc
from dask.dataframe.io.orc.arrow import ArrowORCEngine
from dask.dataframe.io.orc.core import _read_orc

with tempfile.TemporaryDirectory() as tmpdir:
    df = pd.DataFrame({'aa': [0], 'a': [0]})
    table = pa.Table.from_pandas(df)

    path = os.path.join(tmpdir, "test.orc")
    with open(path, 'wb') as f:
        orc.write_table(table, f)

    class FakeFS:
        def open(self, path, mode):
            return open(path, mode)

    fs = FakeFS()
    parts = [(path, None)]
    schema = {'aa': df['aa'].dtype, 'a': df['a'].dtype}

    columns = ['a']
    print(f"Before: columns = {columns}")

    _read_orc(
        parts,
        engine=ArrowORCEngine,
        fs=fs,
        schema=schema,
        index='aa',
        columns=columns
    )

    print(f"After: columns = {columns}")
```

## Why This Is A Bug

Functions should not mutate their input parameters unless explicitly documented. This violates the principle of least surprise and can cause subtle bugs when callers reuse the `columns` list. The mutation happens at line 113 of `core.py` where `columns.append(index)` modifies the input list in place.

## Fix

```diff
--- a/dask/dataframe/io/orc/core.py
+++ b/dask/dataframe/io/orc/core.py
@@ -110,7 +110,8 @@ def read_orc(

 def _read_orc(parts, *, engine, fs, schema, index, columns=None):
     if index is not None and columns is not None:
-        columns.append(index)
+        columns = columns.copy()
+        columns.append(index)
     _df = engine.read_partition(
         fs,
         parts,
```