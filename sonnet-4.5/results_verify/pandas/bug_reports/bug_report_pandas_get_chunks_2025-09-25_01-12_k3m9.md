# Bug Report: pandas.api.interchange get_chunks yields empty chunks

**Target**: `pandas.core.interchange.dataframe.PandasDataFrameXchg.get_chunks`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When `n_chunks` exceeds the number of rows in a DataFrame, `get_chunks()` yields empty chunks at the end, violating the reasonable expectation that chunks should contain data when the DataFrame itself is non-empty.

## Property-Based Test

```python
from hypothesis import given, strategies as st

@given(
    data=st.lists(st.integers(), min_size=1, max_size=20),
    n_chunks=st.integers(min_value=1, max_value=25)
)
def test_get_chunks_no_empty_chunks(data, n_chunks):
    df = pd.DataFrame({'x': data})
    interchange_obj = df.__dataframe__()
    chunks = list(interchange_obj.get_chunks(n_chunks))

    for i, chunk in enumerate(chunks):
        num_rows = chunk.num_rows()
        assert num_rows > 0, f"Chunk {i} is empty (has {num_rows} rows)"
```

**Failing input**: `data=[0]` (1-row DataFrame), `n_chunks=2`

## Reproducing the Bug

```python
import pandas as pd
from pandas.api.interchange import from_dataframe

df = pd.DataFrame({'x': [0]})
interchange_obj = df.__dataframe__()
chunks = list(interchange_obj.get_chunks(n_chunks=2))

for i, chunk in enumerate(chunks):
    print(f"Chunk {i}: {chunk.num_rows()} rows")
```

Output:
```
Chunk 0: 1 rows
Chunk 1: 0 rows
```

## Why This Is A Bug

When a DataFrame has fewer rows than the requested number of chunks, the current implementation creates empty chunks. This violates the reasonable expectation that:
1. Chunks should contain data when the source DataFrame is non-empty
2. Empty chunks serve no useful purpose and waste resources
3. The interchange protocol is designed for efficient data transfer, not empty containers

The bug occurs because the loop at line 107 iterates `n_chunks` times regardless of DataFrame size:
```python
for start in range(0, step * n_chunks, step):
```

When `n_chunks > len(df)`, this creates start positions beyond the DataFrame's bounds, resulting in empty slices.

## Fix

```diff
--- a/pandas/core/interchange/dataframe.py
+++ b/pandas/core/interchange/dataframe.py
@@ -101,10 +101,11 @@ class PandasDataFrameXchg(DataFrameXchg):
         """
         if n_chunks and n_chunks > 1:
             size = len(self._df)
-            step = size // n_chunks
-            if size % n_chunks != 0:
-                step += 1
-            for start in range(0, step * n_chunks, step):
+            # Ensure we don't create more chunks than rows
+            actual_chunks = min(n_chunks, size)
+            step = size // actual_chunks
+            if size % actual_chunks != 0:
+                step += 1
+            for start in range(0, size, step):
                 yield PandasDataFrameXchg(
                     self._df.iloc[start : start + step, :],
                     allow_copy=self._allow_copy,
```

This fix ensures that:
1. We never create more chunks than there are rows
2. The loop only iterates over actual data positions
3. All chunks contain at least one row (when DataFrame is non-empty)