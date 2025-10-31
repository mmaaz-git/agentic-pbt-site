# Bug Report: pandas.api.interchange get_chunks Creates Empty Chunks

**Target**: `pandas.core.interchange.dataframe.PandasDataFrameXchg.get_chunks`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When `n_chunks` exceeds the number of rows in a DataFrame, `get_chunks()` creates empty chunks instead of capping the number of chunks at the row count.

## Property-Based Test

```python
from hypothesis import given, strategies as st


@given(
    st.integers(min_value=1, max_value=100),
    st.integers(min_value=2, max_value=10),
)
def test_chunk_sizes_reasonable(nrows, n_chunks):
    df = pd.DataFrame({'a': list(range(nrows))})
    interchange_df = df.__dataframe__()
    chunks = list(interchange_df.get_chunks(n_chunks=n_chunks))

    chunk_sizes = [chunk.num_rows() for chunk in chunks]

    for size in chunk_sizes:
        assert size > 0, f"Chunk has {size} rows (should be > 0)"
```

**Failing input**: `nrows=1, n_chunks=2`

## Reproducing the Bug

```python
import pandas as pd

df = pd.DataFrame({'a': [0]})
interchange_df = df.__dataframe__()

chunks = list(interchange_df.get_chunks(n_chunks=2))

print(f"Chunk 0: {chunks[0].num_rows()} rows")
print(f"Chunk 1: {chunks[1].num_rows()} rows")
```

**Output:**
```
Chunk 0: 1 rows
Chunk 1: 0 rows
```

## Why This Is A Bug

1. Empty chunks serve no purpose and violate the reasonable expectation that all chunks contain data
2. The protocol documentation doesn't specify behavior when `n_chunks > num_rows`, but creating empty chunks is unintuitive
3. Consumers of the interchange protocol may not expect or handle empty chunks correctly

The bug occurs because the loop on line 107 iterates `n_chunks` times regardless of dataframe size:

```python
for start in range(0, step * n_chunks, step):
```

When `n_chunks=2` and `size=1`:
- `step = 1 // 2 = 0`, then `+= 1` → `step = 1`
- `range(0, 1*2, 1) = [0, 1]` → creates 2 iterations
- Second iteration starts at row 1, which doesn't exist → empty chunk

## Fix

```diff
--- a/pandas/core/interchange/dataframe.py
+++ b/pandas/core/interchange/dataframe.py
@@ -104,7 +104,7 @@ class PandasDataFrameXchg(DataFrameXchg):
             step = size // n_chunks
             if size % n_chunks != 0:
                 step += 1
-            for start in range(0, step * n_chunks, step):
+            for start in range(0, size, step):
                 yield PandasDataFrameXchg(
                     self._df.iloc[start : start + step, :],
                     allow_copy=self._allow_copy,
```

This ensures the loop only iterates over actual rows in the DataFrame, preventing empty chunks.