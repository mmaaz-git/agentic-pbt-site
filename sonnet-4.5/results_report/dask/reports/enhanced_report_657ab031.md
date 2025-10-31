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

        assert read_columns == original_columns, f"Columns were mutated! Before: {original_columns}, After: {read_columns}"

if __name__ == "__main__":
    test_columns_not_mutated_in_read_orc()
```

<details>

<summary>
**Failing input**: `data=[0], columns=['aa', 'a']`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/39/hypo.py", line 55, in <module>
    test_columns_not_mutated_in_read_orc()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/39/hypo.py", line 15, in test_columns_not_mutated_in_read_orc
    @given(

  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/39/hypo.py", line 52, in test_columns_not_mutated_in_read_orc
    assert read_columns == original_columns, f"Columns were mutated! Before: {original_columns}, After: {read_columns}"
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Columns were mutated! Before: ['a'], After: ['a', 'aa']
Falsifying example: test_columns_not_mutated_in_read_orc(
    # The test always failed when commented parts were varied together.
    data=[0],  # or any other generated value
    columns=['aa', 'a'],  # or any other generated value
)
```
</details>

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

<details>

<summary>
Demonstrates unexpected mutation of columns list parameter
</summary>
```
Before: columns = ['a']
After: columns = ['a', 'aa']
```
</details>

## Why This Is A Bug

This bug violates fundamental Python conventions and best practices. Functions should not mutate their input parameters unless:
1. It's explicitly documented in the function's docstring
2. The function name suggests mutation (e.g., `list.sort()` vs `sorted()`)
3. It's the primary purpose of the function

The `_read_orc` function has no documentation indicating it will mutate the `columns` parameter. This unexpected mutation can cause subtle bugs when callers reuse the columns list, leading to:
- Unexpected behavior in subsequent operations
- Difficult-to-debug issues where lists change unexpectedly
- Violation of the principle of least surprise

The mutation occurs at line 113 in `/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages/dask/dataframe/io/orc/core.py` where the code directly calls `columns.append(index)` on the input parameter.

## Relevant Context

The bug affects the call chain from the public API to the internal function:
- `read_orc` (public API, line 32-108) → `dd.from_map` → `_read_orc` (internal, line 111-123)

The public `read_orc` function partially protects against this issue at lines 97-98 when the index is already in the columns list:
```python
if columns is not None and index in columns:
    columns = [col for col in columns if col != index]
```

However, when the index is NOT in the original columns list, the original list is passed directly to `_read_orc` where it gets mutated. This inconsistency makes the behavior unpredictable.

While `_read_orc` is an internal function (indicated by the leading underscore), it's still called through the public API chain with user-provided data, making this mutation affect end users.

## Proposed Fix

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