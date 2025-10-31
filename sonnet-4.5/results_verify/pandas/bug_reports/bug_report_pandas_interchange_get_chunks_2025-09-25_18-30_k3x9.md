# Bug Report: pandas.core.interchange get_chunks Produces Empty Chunks

**Target**: `pandas.core.interchange.dataframe.PandasDataFrameXchg.get_chunks` and `pandas.core.interchange.column.PandasColumn.get_chunks`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `get_chunks` method produces empty chunks when the requested number of chunks cannot evenly divide the DataFrame/Column size. This bug affects both `PandasDataFrameXchg.get_chunks` (dataframe.py:98-113) and `PandasColumn.get_chunks` (column.py:250-265).

## Property-Based Test

```python
import pandas as pd
from hypothesis import given, strategies as st, settings
from pandas.core.interchange.dataframe import PandasDataFrameXchg


@given(
    size=st.integers(min_value=1, max_value=100),
    n_chunks=st.integers(min_value=2, max_value=20)
)
@settings(max_examples=500)
def test_get_chunks_should_not_produce_empty_chunks(size, n_chunks):
    df = pd.DataFrame({'A': range(size)})
    interchange_obj = PandasDataFrameXchg(df)

    chunks = list(interchange_obj.get_chunks(n_chunks))

    for i, chunk in enumerate(chunks):
        if chunk.num_rows() == 0:
            raise AssertionError(
                f"Chunk {i} is empty! size={size}, n_chunks={n_chunks}, "
                f"chunk_sizes={[c.num_rows() for c in chunks]}"
            )
```

**Failing input**: `size=1, n_chunks=2`

## Reproducing the Bug

```python
import pandas as pd
from pandas.core.interchange.dataframe import PandasDataFrameXchg

df = pd.DataFrame({'A': range(11)})
interchange_obj = PandasDataFrameXchg(df)
chunks = list(interchange_obj.get_chunks(5))

print(f"Chunk sizes: {[chunk.num_rows() for chunk in chunks]}")
```

Output:
```
Chunk sizes: [3, 3, 3, 2, 0]
```

The last chunk is empty. The same bug occurs with `PandasColumn.get_chunks`:

```python
import pandas as pd
from pandas.core.interchange.column import PandasColumn

series = pd.Series(range(11))
column = PandasColumn(series)
chunks = list(column.get_chunks(5))

print(f"Chunk sizes: {[chunk.size() for chunk in chunks]}")
```

Output:
```
Chunk sizes: [3, 3, 3, 2, 0]
```

## Why This Is A Bug

When splitting a DataFrame or Column into chunks, all chunks should contain at least some data. Empty chunks are useless and can cause issues in downstream processing. The issue occurs because the chunking algorithm incorrectly calculates the range to iterate over, extending beyond the DataFrame/Column's actual size.

The flawed logic:
```python
step = size // n_chunks
if size % n_chunks != 0:
    step += 1
for start in range(0, step * n_chunks, step):  # Bug: step * n_chunks can exceed size
    ...
```

For example, with size=11 and n_chunks=5:
- step = 11 // 5 = 2
- step becomes 3 (due to remainder)
- range(0, 15, 3) = [0, 3, 6, 9, 12]
- But the DataFrame only has indices 0-10, so slicing [12:15] produces an empty chunk

## Fix

The same fix applies to both files - change `range(0, step * n_chunks, step)` to `range(0, size, step)`:

```diff
--- a/pandas/core/interchange/dataframe.py
+++ b/pandas/core/interchange/dataframe.py
@@ -98,11 +98,11 @@ class PandasDataFrameXchg(DataFrameXchg):
     def get_chunks(self, n_chunks: int | None = None) -> Iterable[PandasDataFrameXchg]:
         """
         Return an iterator yielding the chunks.
         """
         if n_chunks and n_chunks > 1:
             size = len(self._df)
             step = size // n_chunks
             if size % n_chunks != 0:
                 step += 1
-            for start in range(0, step * n_chunks, step):
+            for start in range(0, size, step):
                 yield PandasDataFrameXchg(
                     self._df.iloc[start : start + step, :],
                     allow_copy=self._allow_copy,
                 )
         else:
             yield self
```

```diff
--- a/pandas/core/interchange/column.py
+++ b/pandas/core/interchange/column.py
@@ -250,11 +250,11 @@ class PandasColumn(Column):
     def get_chunks(self, n_chunks: int | None = None):
         """
         Return an iterator yielding the chunks.
         See `DataFrame.get_chunks` for details on ``n_chunks``.
         """
         if n_chunks and n_chunks > 1:
             size = len(self._col)
             step = size // n_chunks
             if size % n_chunks != 0:
                 step += 1
-            for start in range(0, step * n_chunks, step):
+            for start in range(0, size, step):
                 yield PandasColumn(
                     self._col.iloc[start : start + step], self._allow_copy
                 )
         else:
             yield self
```

This fix ensures we only iterate over the actual size of the DataFrame/Column, preventing empty chunks from being created.