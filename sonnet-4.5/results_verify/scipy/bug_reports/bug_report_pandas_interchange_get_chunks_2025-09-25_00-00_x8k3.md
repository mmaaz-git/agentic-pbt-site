# Bug Report: pandas.core.interchange get_chunks() Creates Empty Chunks

**Target**: `pandas.core.interchange.dataframe.PandasDataFrameXchg.get_chunks()` and `pandas.core.interchange.column.PandasColumn.get_chunks()`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

When `get_chunks(n_chunks)` is called with `n_chunks` greater than the number of rows/elements in the data, the method creates empty chunks instead of returning a reasonable number of non-empty chunks. This violates user expectations and creates wasteful iteration.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pandas as pd
import numpy as np
from pandas.core.interchange.dataframe import PandasDataFrameXchg

@given(
    n_rows=st.integers(min_value=1, max_value=100),
    n_chunks=st.integers(min_value=2, max_value=200),
)
def test_no_empty_chunks(n_rows, n_chunks):
    df = pd.DataFrame(np.random.randn(n_rows, 3))
    xchg = PandasDataFrameXchg(df, allow_copy=True)
    chunks = list(xchg.get_chunks(n_chunks=n_chunks))

    empty_chunks = [c for c in chunks if c.num_rows() == 0]
    assert len(empty_chunks) == 0, f"Found {len(empty_chunks)} empty chunks"
```

**Failing input**: `n_rows=5, n_chunks=7`

## Reproducing the Bug

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({'A': [0, 1, 2, 3, 4]})
interchange_df = df.__dataframe__()
chunks = list(interchange_df.get_chunks(n_chunks=7))

print(f"DataFrame has {len(df)} rows")
print(f"Requested 7 chunks, got {len(chunks)} chunks")

for i, chunk in enumerate(chunks):
    num_rows = chunk.num_rows()
    print(f"Chunk {i}: {num_rows} rows")

empty_count = sum(1 for c in chunks if c.num_rows() == 0)
print(f"Empty chunks: {empty_count}")
```

**Output:**
```
DataFrame has 5 rows
Requested 7 chunks, got 7 chunks
Chunk 0: 1 rows
Chunk 1: 1 rows
Chunk 2: 1 rows
Chunk 3: 1 rows
Chunk 4: 1 rows
Chunk 5: 0 rows
Chunk 6: 0 rows
Empty chunks: 2
```

## Why This Is A Bug

The interchange protocol documentation states that `get_chunks()` should "subdivide each chunk" when `n_chunks` is specified. Subdividing existing data should not create empty placeholders. When a user requests 7 chunks of a 5-row DataFrame, reasonable behaviors would be:

1. Return 5 chunks (one per row) - the maximum meaningful subdivision
2. Raise an error indicating n_chunks is too large
3. Silently cap at the data size

Creating empty chunks is problematic because:
- It violates user expectations - users don't expect to iterate over empty data
- It's wasteful - downstream code must handle empty chunks unnecessarily
- It could cause bugs in consumer code that doesn't anticipate empty chunks
- The same data with different chunk requests yields different chunk counts, which is confusing

## Fix

```diff
--- a/pandas/core/interchange/dataframe.py
+++ b/pandas/core/interchange/dataframe.py
@@ -99,9 +99,11 @@ class PandasDataFrameXchg(DataFrameXchg):
     def get_chunks(self, n_chunks: int | None = None) -> Iterable[PandasDataFrameXchg]:
         """
         Return an iterator yielding the chunks.
         """
         if n_chunks and n_chunks > 1:
             size = len(self._df)
+            # Don't create more chunks than we have rows
+            n_chunks = min(n_chunks, size)
             step = size // n_chunks
             if size % n_chunks != 0:
                 step += 1
             for start in range(0, step * n_chunks, step):
```

The same fix should be applied to `pandas/core/interchange/column.py` at line 256:

```diff
--- a/pandas/core/interchange/column.py
+++ b/pandas/core/interchange/column.py
@@ -253,6 +253,8 @@ class PandasColumn(Column):
         See `DataFrame.get_chunks` for details on ``n_chunks``.
         """
         if n_chunks and n_chunks > 1:
             size = len(self._col)
+            # Don't create more chunks than we have elements
+            n_chunks = min(n_chunks, size)
             step = size // n_chunks
             if size % n_chunks != 0:
                 step += 1
```