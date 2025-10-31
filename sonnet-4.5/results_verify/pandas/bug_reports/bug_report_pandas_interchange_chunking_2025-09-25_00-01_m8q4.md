# Bug Report: pandas.core.interchange Empty DataFrame/Column Chunking Crash

**Target**: `pandas.core.interchange.dataframe.PandasDataFrameXchg.get_chunks` and `pandas.core.interchange.column.PandasColumn.get_chunks`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `get_chunks` method crashes with `ValueError: range() arg 3 must not be zero` when attempting to chunk an empty DataFrame or Series with `n_chunks > 1`. This bug affects both `PandasDataFrameXchg` and `PandasColumn` classes.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import pandas as pd
from pandas.core.interchange.dataframe import PandasDataFrameXchg


@settings(max_examples=500)
@given(
    size=st.integers(min_value=0, max_value=20),
    n_chunks=st.integers(min_value=1, max_value=10)
)
def test_chunking_never_crashes(size, n_chunks):
    df = pd.DataFrame({'A': list(range(size))})
    interchange_df = PandasDataFrameXchg(df)

    chunks = list(interchange_df.get_chunks(n_chunks))

    total_rows = sum(chunk.num_rows() for chunk in chunks)
    assert total_rows == size
```

**Failing input**: `size=0, n_chunks=2` (or any n_chunks > 1)

## Reproducing the Bug

**For DataFrame:**
```python
import pandas as pd
from pandas.core.interchange.dataframe import PandasDataFrameXchg

df = pd.DataFrame({'A': []})
interchange_df = PandasDataFrameXchg(df)

chunks = list(interchange_df.get_chunks(2))
```

**For Column:**
```python
import pandas as pd
from pandas.core.interchange.column import PandasColumn

series = pd.Series([])
col = PandasColumn(series)

chunks = list(col.get_chunks(2))
```

Both produce:
```
ValueError: range() arg 3 must not be zero
```

## Why This Is A Bug

1. Empty DataFrames and Series are valid and commonly occur in data processing
2. The interchange protocol should handle edge cases gracefully
3. The error occurs because when `size=0` and `n_chunks=2`:
   - `step = 0 // 2 = 0`
   - `size % n_chunks = 0 % 2 = 0` (so step doesn't increment)
   - `range(0, 0 * 2, 0) = range(0, 0, 0)` → ValueError
4. The expected behavior is to return either an empty iterator or a single empty chunk

## Fix

The same fix is needed in both files:

**For dataframe.py:**
```diff
--- a/pandas/core/interchange/dataframe.py
+++ b/pandas/core/interchange/dataframe.py
@@ -102,6 +102,9 @@ class PandasDataFrameXchg(DataFrameXchg):
         if n_chunks and n_chunks > 1:
             size = len(self._df)
             step = size // n_chunks
+            if step == 0:
+                yield self
+                return
             if size % n_chunks != 0:
                 step += 1
             for start in range(0, step * n_chunks, step):
```

**For column.py:**
```diff
--- a/pandas/core/interchange/column.py
+++ b/pandas/core/interchange/column.py
@@ -255,6 +255,9 @@ class PandasColumn(Column):
         if n_chunks and n_chunks > 1:
             size = len(self._col)
             step = size // n_chunks
+            if step == 0:
+                yield self
+                return
             if size % n_chunks != 0:
                 step += 1
             for start in range(0, step * n_chunks, step):
```

This fix ensures that when the DataFrame/Series is too small to be meaningfully split (including when it's empty), the method yields the entire object as a single chunk rather than crashing.