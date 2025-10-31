# Bug Report: pandas.core.interchange get_chunks Creates Empty Chunks

**Target**: `pandas.core.interchange.dataframe.PandasDataFrameXchg.get_chunks`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `get_chunks()` method in `PandasDataFrameXchg` creates empty chunks when `n_chunks >= n_rows`, violating the reasonable expectation that chunking should not produce empty data chunks.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

import pandas as pd
from hypothesis import given, strategies as st, settings
from pandas.core.interchange.dataframe import PandasDataFrameXchg


@given(st.data())
@settings(max_examples=200)
def test_get_chunks_no_empty_chunks(data):
    n_rows = data.draw(st.integers(min_value=1, max_value=100))

    df = pd.DataFrame({'col': list(range(n_rows))})
    xchg = PandasDataFrameXchg(df)

    n_chunks = data.draw(st.integers(min_value=1, max_value=20))

    chunks = list(xchg.get_chunks(n_chunks))

    for i, chunk in enumerate(chunks):
        assert chunk.num_rows() > 0, f"Chunk {i} is empty (n_rows={n_rows}, n_chunks={n_chunks})"
```

**Failing input**: `n_rows=1, n_chunks=2`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

import pandas as pd
from pandas.core.interchange.dataframe import PandasDataFrameXchg

df = pd.DataFrame({'col': [0]})
xchg = PandasDataFrameXchg(df)
chunks = list(xchg.get_chunks(2))

print(f"Chunk 0: {chunks[0].num_rows()} rows")
print(f"Chunk 1: {chunks[1].num_rows()} rows")
```

**Output:**
```
Chunk 0: 1 rows
Chunk 1: 0 rows
```

## Why This Is A Bug

When chunking a DataFrame, users expect all chunks to contain data. Empty chunks are problematic because:

1. **Protocol semantics**: The DataFrame interchange protocol should return meaningful chunks of data, not empty placeholders
2. **Downstream failures**: Code consuming chunks may not handle empty chunks correctly, leading to crashes or incorrect results
3. **Resource waste**: Creating and iterating over empty chunks wastes computational resources
4. **Unexpected behavior**: When requesting `n_chunks`, users expect approximately equal-sized chunks of actual data, not `n_chunks` iterations where some are empty

The root cause is in line 107 of `dataframe.py`: the loop uses `range(0, step * n_chunks, step)`, which can exceed the actual data size when `n_chunks > n_rows`.

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

This fix ensures the loop only iterates over actual data indices, preventing the creation of empty chunks.