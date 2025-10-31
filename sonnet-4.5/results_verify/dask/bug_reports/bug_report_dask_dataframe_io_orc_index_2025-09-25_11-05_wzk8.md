# Bug Report: dask.dataframe.io.orc Index Corruption in Multi-Partition Round-Trip

**Target**: `dask.dataframe.io.orc.read_orc` and `dask.dataframe.io.orc.to_orc`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When writing a multi-partition Dask DataFrame to ORC without the index (`write_index=False`) and reading it back, the resulting index contains duplicate values instead of a sequential RangeIndex, causing silent data corruption.

## Property-Based Test

```python
import tempfile
import pandas as pd
import dask.dataframe as dd
from hypothesis import given, strategies as st, settings

@given(
    st.lists(st.integers(), min_size=1, max_size=100),
    st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=1, max_size=100),
)
@settings(max_examples=50)
def test_round_trip_without_index(int_data, float_data):
    min_len = min(len(int_data), len(float_data))
    df_pandas = pd.DataFrame({
        'x': int_data[:min_len],
        'y': float_data[:min_len],
    })
    df = dd.from_pandas(df_pandas, npartitions=2)

    with tempfile.TemporaryDirectory() as tmpdir:
        dd.io.orc.to_orc(df, tmpdir, write_index=False)
        df_loaded = dd.io.orc.read_orc(tmpdir)
        df_result = df_loaded.compute()
        df_expected = df.compute().reset_index(drop=True)

        pd.testing.assert_frame_equal(df_result, df_expected)
```

**Failing input**: `int_data=[0, 0], float_data=[0.0, 0.0]`

## Reproducing the Bug

```python
import tempfile
import pandas as pd
import dask.dataframe as dd

df_pandas = pd.DataFrame({
    'x': [0, 0],
    'y': [0.0, 0.0],
})
df = dd.from_pandas(df_pandas, npartitions=2)

with tempfile.TemporaryDirectory() as tmpdir:
    dd.io.orc.to_orc(df, tmpdir, write_index=False)
    df_loaded = dd.io.orc.read_orc(tmpdir)
    df_result = df_loaded.compute()

print("Original index:", list(df.compute().index))
print("Loaded index:", list(df_result.index))
```

**Output:**
```
Original index: [0, 1]
Loaded index: [0, 0]
```

## Why This Is A Bug

When `write_index=False`, each partition is written to a separate ORC file with its local RangeIndex starting from 0. When reading back these files, Dask does not reconstruct a global sequential index; instead, it preserves the per-partition local indices, resulting in duplicate index values across partitions. This violates the fundamental expectation that a round-trip operation should preserve the data structure, and creates a dataframe with a malformed index that could lead to incorrect results in subsequent operations.

## Fix

The issue is that when reading ORC files without an explicit index column, Dask should either:
1. Reset the index to create a global sequential RangeIndex, or
2. Adjust partition-local indices to be globally unique

A simple fix would be to reset the index in `read_orc` when no explicit index is specified:

```diff
--- a/dask/dataframe/io/orc/__init__.py
+++ b/dask/dataframe/io/orc/__init__.py
@@ -66,10 +66,15 @@ def read_orc(

     if columns is not None and index in columns:
         columns = [col for col in columns if col != index]
-    return dd.from_map(
+
+    result = dd.from_map(
         _read_orc,
         parts,
         engine=engine,
         fs=fs,
         schema=schema,
         index=index,
         meta=meta,
         columns=columns,
     )
+
+    if index is None:
+        result = result.reset_index(drop=True)
+
+    return result
```