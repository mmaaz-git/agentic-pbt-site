# Bug Report: pandas.core.interchange Get Chunks Creates Empty Chunks

**Target**: `pandas.core.interchange.dataframe.PandasDataFrameXchg.get_chunks`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `get_chunks` method creates empty chunks when requested chunk count exceeds the number of rows in the DataFrame. This wastes computation resources and violates the reasonable expectation that chunks should contain data.

## Property-Based Test

```python
import pandas as pd
from hypothesis import given, strategies as st
from pandas.core.interchange.dataframe import PandasDataFrameXchg


@given(
    n_rows=st.integers(min_value=1, max_value=100),
    n_chunks=st.integers(min_value=2, max_value=20)
)
def test_get_chunks_no_empty_chunks(n_rows, n_chunks):
    df = pd.DataFrame({'a': range(n_rows)})
    xchg_df = PandasDataFrameXchg(df)
    chunks = list(xchg_df.get_chunks(n_chunks))

    for i, chunk in enumerate(chunks):
        assert chunk.num_rows() > 0, f"Chunk {i} is empty"
```

**Failing input**: `n_rows=1, n_chunks=2`

## Reproducing the Bug

```python
import pandas as pd
from pandas.core.interchange.dataframe import PandasDataFrameXchg

df = pd.DataFrame({'a': [1]})
xchg_df = PandasDataFrameXchg(df)

chunks = list(xchg_df.get_chunks(n_chunks=2))

for i, chunk in enumerate(chunks):
    print(f"Chunk {i}: {chunk.num_rows()} rows")
```

Output:
```
Chunk 0: 1 rows
Chunk 1: 0 rows
```

The second chunk is empty. Similarly, with 5 rows and 10 chunks, you get 5 chunks with 1 row each and 5 empty chunks.

## Why This Is A Bug

1. Empty chunks provide no data and waste computational resources
2. The function creates exactly `n_chunks` chunks even when this results in empty chunks
3. When `n_chunks > n_rows`, it should return at most `n_rows` chunks (one per row)
4. Downstream code may not expect empty chunks and could behave unexpectedly
5. The chunking logic uses `step * n_chunks` as the upper bound in the range, which causes iteration beyond the data

The root cause is in this code:
```python
step = size // n_chunks
if size % n_chunks != 0:
    step += 1
for start in range(0, step * n_chunks, step):  # Bug: step * n_chunks can exceed size
    yield PandasDataFrameXchg(
        self._df.iloc[start : start + step, :],
        allow_copy=self._allow_copy,
    )
```

## Fix

Fix the chunking logic to never create empty chunks:

```diff
diff --git a/pandas/core/interchange/dataframe.py b/pandas/core/interchange/dataframe.py
index 1234567..abcdefg 100644
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

This changes the range upper bound from `step * n_chunks` to `size`, ensuring we never iterate beyond the actual data.