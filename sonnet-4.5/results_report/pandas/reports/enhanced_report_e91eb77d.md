# Bug Report: pandas.core.interchange.dataframe.PandasDataFrameXchg.get_chunks Creates Empty Chunks When n_chunks Exceeds Row Count

**Target**: `pandas.core.interchange.dataframe.PandasDataFrameXchg.get_chunks`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `get_chunks` method in pandas interchange protocol creates empty DataFrame chunks when the requested number of chunks exceeds the number of rows in the DataFrame, wasting computational resources and violating reasonable expectations about chunking behavior.

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

# Run the test
if __name__ == "__main__":
    test_get_chunks_no_empty_chunks()
```

<details>

<summary>
**Failing input**: `n_rows=1, n_chunks=2`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/56/hypo.py", line 20, in <module>
    test_get_chunks_no_empty_chunks()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/56/hypo.py", line 7, in test_get_chunks_no_empty_chunks
    n_rows=st.integers(min_value=1, max_value=100),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/56/hypo.py", line 16, in test_get_chunks_no_empty_chunks
    assert chunk.num_rows() > 0, f"Chunk {i} is empty"
           ^^^^^^^^^^^^^^^^^^^^
AssertionError: Chunk 1 is empty
Falsifying example: test_get_chunks_no_empty_chunks(
    n_rows=1,
    n_chunks=2,  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
import pandas as pd
from pandas.core.interchange.dataframe import PandasDataFrameXchg

# Test case 1: 1 row, 2 chunks requested
print("Test case 1: 1 row, 2 chunks requested")
df = pd.DataFrame({'a': [1]})
xchg_df = PandasDataFrameXchg(df)

chunks = list(xchg_df.get_chunks(n_chunks=2))

for i, chunk in enumerate(chunks):
    print(f"Chunk {i}: {chunk.num_rows()} rows")

print("\n" + "="*50 + "\n")

# Test case 2: 5 rows, 10 chunks requested
print("Test case 2: 5 rows, 10 chunks requested")
df2 = pd.DataFrame({'a': range(5)})
xchg_df2 = PandasDataFrameXchg(df2)

chunks2 = list(xchg_df2.get_chunks(n_chunks=10))

for i, chunk in enumerate(chunks2):
    print(f"Chunk {i}: {chunk.num_rows()} rows")
```

<details>

<summary>
Empty chunks created when n_chunks > num_rows
</summary>
```
Test case 1: 1 row, 2 chunks requested
Chunk 0: 1 rows
Chunk 1: 0 rows

==================================================

Test case 2: 5 rows, 10 chunks requested
Chunk 0: 1 rows
Chunk 1: 1 rows
Chunk 2: 1 rows
Chunk 3: 1 rows
Chunk 4: 1 rows
Chunk 5: 0 rows
Chunk 6: 0 rows
Chunk 7: 0 rows
Chunk 8: 0 rows
Chunk 9: 0 rows
```
</details>

## Why This Is A Bug

This behavior violates reasonable expectations about chunking functionality. While the DataFrame Interchange Protocol documentation (dataframe_protocol.py:457-465) doesn't explicitly forbid empty chunks, creating chunks with no data serves no practical purpose and wastes computational resources. The method always returns exactly `n_chunks` chunks regardless of available data, resulting in empty DataFrames when `n_chunks > num_rows`.

The bug stems from a logic error in the chunking loop that iterates beyond the DataFrame's actual size. When calculating the range bounds, the code uses `step * n_chunks` which can exceed the DataFrame size, causing `iloc` slicing to return empty DataFrames for indices beyond the data bounds. This behavior is inconsistent with the principle that chunks should partition the available data, not create phantom empty partitions.

## Relevant Context

The bug occurs in `/pandas/core/interchange/dataframe.py` at line 107 within the `get_chunks` method. The pandas DataFrame Interchange Protocol is a relatively new feature designed for cross-library data exchange, and pandas documentation already notes "severe implementation issues" with this protocol, recommending the Arrow C Data Interface as an alternative.

The issue manifests when users request more chunks than there are rows in the DataFrame. For instance:
- A DataFrame with 1 row split into 2 chunks produces 1 chunk with data and 1 empty chunk
- A DataFrame with 5 rows split into 10 chunks produces 5 chunks with 1 row each and 5 empty chunks

This pattern could cause unexpected behavior in downstream code that assumes chunks contain data, potentially leading to wasted computation or logic errors in data processing pipelines.

Relevant code location: https://github.com/pandas-dev/pandas/blob/main/pandas/core/interchange/dataframe.py#L98-L114

## Proposed Fix

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