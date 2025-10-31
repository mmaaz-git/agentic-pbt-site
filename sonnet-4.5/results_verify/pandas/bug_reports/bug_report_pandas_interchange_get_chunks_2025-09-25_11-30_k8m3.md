# Bug Report: pandas.api.interchange get_chunks produces empty chunks

**Target**: `pandas.core.interchange.dataframe.PandasDataFrameXchg.get_chunks` and `pandas.core.interchange.column.PandasColumn.get_chunks`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When `get_chunks(n_chunks)` is called with `n_chunks` greater than the number of rows in the DataFrame, it produces empty chunks, which violates the expectation that chunks contain actual data.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st
import pandas as pd

@given(
    st.integers(min_value=1, max_value=50),
    st.integers(min_value=1, max_value=50)
)
@settings(max_examples=1000)
def test_chunks_no_empty_chunks(size, n_chunks):
    df = pd.DataFrame({'A': range(size)})
    interchange_obj = df.__dataframe__()

    chunks = list(interchange_obj.get_chunks(n_chunks=n_chunks))

    num_empty_chunks = sum(1 for chunk in chunks if chunk.num_rows() == 0)

    assert num_empty_chunks == 0, \
        f"Got {num_empty_chunks} empty chunks for size={size} with n_chunks={n_chunks}"
```

**Failing input**: `size=1, n_chunks=2` (minimal example found by Hypothesis)

## Reproducing the Bug

```python
import pandas as pd

df = pd.DataFrame({'A': [1, 2, 3]})
interchange_obj = df.__dataframe__()

chunks = list(interchange_obj.get_chunks(n_chunks=4))

print("DataFrame size:", len(df))
print("Requested chunks:", 4)
print("Chunks generated:", len(chunks))
print("Chunk sizes:", [chunk.num_rows() for chunk in chunks])
```

Output:
```
DataFrame size: 3
Requested chunks: 4
Chunks generated: 4
Chunk sizes: [1, 1, 1, 0]
```

## Why This Is A Bug

1. Empty chunks serve no purpose and waste resources for consumers iterating over chunks
2. Consumers reasonably expect that all chunks contain data
3. The DataFrame Interchange Protocol specification doesn't require producing empty chunks
4. Empty chunks can cause issues in downstream processing where chunk size is assumed to be > 0

The bug occurs in the chunking algorithm in both `/pandas/core/interchange/dataframe.py` and `/pandas/core/interchange/column.py` (identical implementation):

```python
def get_chunks(self, n_chunks: int | None = None) -> Iterable[PandasDataFrameXchg]:
    if n_chunks and n_chunks > 1:
        size = len(self._df)
        step = size // n_chunks
        if size % n_chunks != 0:
            step += 1
        for start in range(0, step * n_chunks, step):
            yield PandasDataFrameXchg(
                self._df.iloc[start : start + step, :],
                allow_copy=self._allow_copy,
            )
    else:
        yield self
```

When `n_chunks > size`:
- For size=1, n_chunks=2: step=1, range(0,2,1)=[0,1], yields iloc[0:1] and iloc[1:2] (empty)
- The algorithm always generates exactly `n_chunks` chunks, even when insufficient rows exist

## Fix

The fix needs to be applied to both `dataframe.py` and `column.py`:

```diff
--- a/pandas/core/interchange/dataframe.py
+++ b/pandas/core/interchange/dataframe.py
@@ -100,11 +100,12 @@ class PandasDataFrameXchg(DataFrameXchg):
         """
         if n_chunks and n_chunks > 1:
             size = len(self._df)
+            n_chunks = min(n_chunks, size)
             step = size // n_chunks
             if size % n_chunks != 0:
                 step += 1
             for start in range(0, step * n_chunks, step):
+                if start >= size:
+                    break
                 yield PandasDataFrameXchg(
                     self._df.iloc[start : start + step, :],
                     allow_copy=self._allow_copy,
```

And similarly for `column.py`:

```diff
--- a/pandas/core/interchange/column.py
+++ b/pandas/core/interchange/column.py
@@ -254,11 +254,12 @@ class PandasColumn(Column):
         """
         if n_chunks and n_chunks > 1:
             size = len(self._col)
+            n_chunks = min(n_chunks, size)
             step = size // n_chunks
             if size % n_chunks != 0:
                 step += 1
             for start in range(0, step * n_chunks, step):
+                if start >= size:
+                    break
                 yield PandasColumn(
                     self._col.iloc[start : start + step], self._allow_copy
                 )
```

This fix ensures that:
1. We never request more chunks than rows/elements (`n_chunks = min(n_chunks, size)`)
2. We stop iterating once we've exhausted all data (`if start >= size: break`)