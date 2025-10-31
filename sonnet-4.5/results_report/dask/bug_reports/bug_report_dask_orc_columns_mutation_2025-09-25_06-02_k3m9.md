# Bug Report: dask.dataframe.io.orc._read_orc Mutates Input Columns List

**Target**: `dask.dataframe.io.orc.core._read_orc`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `_read_orc` function mutates its input `columns` list by appending the `index` parameter to it, violating the principle that functions should not mutate their inputs unless explicitly documented.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from dask.dataframe.io.orc.core import _read_orc
from dask.dataframe.io.orc.arrow import ArrowORCEngine


@given(
    index=st.one_of(st.none(), st.text(min_size=1)),
    columns=st.one_of(st.none(), st.lists(st.text(min_size=1), min_size=1, max_size=5))
)
@settings(max_examples=100)
def test_read_orc_does_not_mutate_columns(index, columns):
    if columns is not None:
        columns_before = list(columns)

        try:
            _read_orc(
                parts=[],
                engine=ArrowORCEngine,
                fs=None,
                schema={},
                index=index,
                columns=columns
            )
        except:
            pass

        assert columns == columns_before, f"columns was mutated from {columns_before} to {columns}"
```

**Failing input**: `index='0', columns=['0']`

## Reproducing the Bug

```python
from dask.dataframe.io.orc.core import _read_orc
from dask.dataframe.io.orc.arrow import ArrowORCEngine

columns = ['col1', 'col2']
print(f"Before: {columns}")

try:
    _read_orc(
        parts=[],
        engine=ArrowORCEngine,
        fs=None,
        schema={},
        index='col1',
        columns=columns
    )
except:
    pass

print(f"After: {columns}")
```

Output:
```
Before: ['col1', 'col2']
After: ['col1', 'col2', 'col1']
```

## Why This Is A Bug

The function modifies the input `columns` list by appending `index` to it (line 113 in `core.py`). This violates Python's principle of least surprise - callers don't expect their input lists to be modified. This can cause subtle bugs when the same `columns` list is reused across multiple function calls.

## Fix

```diff
--- a/dask/dataframe/io/orc/core.py
+++ b/dask/dataframe/io/orc/core.py
@@ -110,7 +110,8 @@ def to_orc(

 def _read_orc(parts, *, engine, fs, schema, index, columns=None):
     if index is not None and columns is not None:
-        columns.append(index)
+        columns = columns.copy()
+        columns.append(index)
     _df = engine.read_partition(
         fs,
         parts,
```