# Bug Report: pandas.core.interchange get_chunks Produces Empty Chunks

**Target**: `pandas.core.interchange.dataframe.PandasDataFrameXchg.get_chunks` and `pandas.core.interchange.column.PandasColumn.get_chunks`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `get_chunks` method in pandas interchange protocol produces empty chunks when the requested number of chunks cannot evenly divide the DataFrame/Column size, violating the expected behavior of data chunking operations.

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


if __name__ == "__main__":
    # Run the test
    test_get_chunks_should_not_produce_empty_chunks()
```

<details>

<summary>
**Failing input**: `size=1, n_chunks=2`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/2/hypo.py", line 27, in <module>
    test_get_chunks_should_not_produce_empty_chunks()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/2/hypo.py", line 7, in test_get_chunks_should_not_produce_empty_chunks
    size=st.integers(min_value=1, max_value=100),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/2/hypo.py", line 19, in test_get_chunks_should_not_produce_empty_chunks
    raise AssertionError(
    ...<2 lines>...
    )
AssertionError: Chunk 1 is empty! size=1, n_chunks=2, chunk_sizes=[1, 0]
Falsifying example: test_get_chunks_should_not_produce_empty_chunks(
    size=1,
    n_chunks=2,
)
```
</details>

## Reproducing the Bug

```python
import pandas as pd
from pandas.core.interchange.dataframe import PandasDataFrameXchg
from pandas.core.interchange.column import PandasColumn

# Test case 1: DataFrame with 11 rows split into 5 chunks
print("Test 1: DataFrame with 11 rows split into 5 chunks")
print("-" * 50)
df = pd.DataFrame({'A': range(11)})
interchange_obj = PandasDataFrameXchg(df)
chunks = list(interchange_obj.get_chunks(5))

print(f"DataFrame size: {len(df)} rows")
print(f"Number of chunks requested: 5")
print(f"Chunk sizes: {[chunk.num_rows() for chunk in chunks]}")
print(f"Total rows in all chunks: {sum(chunk.num_rows() for chunk in chunks)}")
for i, chunk in enumerate(chunks):
    if chunk.num_rows() == 0:
        print(f"WARNING: Chunk {i} is EMPTY!")

# Test case 2: Minimal failing case - size=1, n_chunks=2
print("\nTest 2: Minimal failing case - size=1, n_chunks=2")
print("-" * 50)
df_small = pd.DataFrame({'A': [0]})
interchange_obj_small = PandasDataFrameXchg(df_small)
chunks_small = list(interchange_obj_small.get_chunks(2))

print(f"DataFrame size: {len(df_small)} rows")
print(f"Number of chunks requested: 2")
print(f"Chunk sizes: {[chunk.num_rows() for chunk in chunks_small]}")
for i, chunk in enumerate(chunks_small):
    if chunk.num_rows() == 0:
        print(f"WARNING: Chunk {i} is EMPTY!")

# Test case 3: Column version of the same bug
print("\nTest 3: Column version - 11 rows split into 5 chunks")
print("-" * 50)
series = pd.Series(range(11))
column = PandasColumn(series)
col_chunks = list(column.get_chunks(5))

print(f"Series size: {len(series)} rows")
print(f"Number of chunks requested: 5")
print(f"Chunk sizes: {[chunk.size() for chunk in col_chunks]}")
for i, chunk in enumerate(col_chunks):
    if chunk.size() == 0:
        print(f"WARNING: Chunk {i} is EMPTY!")

# Test case 4: Edge case - more chunks than rows
print("\nTest 4: Edge case - 3 rows split into 5 chunks")
print("-" * 50)
df_edge = pd.DataFrame({'A': [1, 2, 3]})
interchange_obj_edge = PandasDataFrameXchg(df_edge)
chunks_edge = list(interchange_obj_edge.get_chunks(5))

print(f"DataFrame size: {len(df_edge)} rows")
print(f"Number of chunks requested: 5")
print(f"Chunk sizes: {[chunk.num_rows() for chunk in chunks_edge]}")
empty_chunks = sum(1 for chunk in chunks_edge if chunk.num_rows() == 0)
print(f"Number of empty chunks: {empty_chunks}")
```

<details>

<summary>
Output demonstrating empty chunks
</summary>
```
Test 1: DataFrame with 11 rows split into 5 chunks
--------------------------------------------------
DataFrame size: 11 rows
Number of chunks requested: 5
Chunk sizes: [3, 3, 3, 2, 0]
Total rows in all chunks: 11
WARNING: Chunk 4 is EMPTY!

Test 2: Minimal failing case - size=1, n_chunks=2
--------------------------------------------------
DataFrame size: 1 rows
Number of chunks requested: 2
Chunk sizes: [1, 0]
WARNING: Chunk 1 is EMPTY!

Test 3: Column version - 11 rows split into 5 chunks
--------------------------------------------------
Series size: 11 rows
Number of chunks requested: 5
Chunk sizes: [3, 3, 3, 2, 0]
WARNING: Chunk 4 is EMPTY!

Test 4: Edge case - 3 rows split into 5 chunks
--------------------------------------------------
DataFrame size: 3 rows
Number of chunks requested: 5
Chunk sizes: [1, 1, 1, 0, 0]
Number of empty chunks: 2
```
</details>

## Why This Is A Bug

This behavior violates expected chunking semantics for several reasons:

1. **Violates Mathematical Expectations**: When dividing N items into K chunks, the expectation is that each chunk contains at least ⌊N/K⌋ items. Empty chunks serve no purpose in data processing.

2. **Wastes Computational Resources**: Empty chunks consume resources in parallel processing pipelines without contributing any useful work. Systems that spawn workers per chunk will create unnecessary overhead.

3. **Can Cause Downstream Errors**: Code consuming these chunks may assume non-empty data and fail when encountering empty chunks. For example, aggregation functions may error on empty DataFrames.

4. **Inconsistent with Protocol Purpose**: The dataframe interchange protocol is designed for efficient data exchange. Empty chunks don't contribute to this goal and complicate consumer implementations.

5. **Algorithm Logic Error**: The bug stems from an incorrect range calculation in the implementation. The code iterates to `step * n_chunks` instead of `size`, causing it to create chunks beyond the actual data bounds.

## Relevant Context

The bug occurs in both `PandasDataFrameXchg.get_chunks` (pandas/core/interchange/dataframe.py:98-113) and `PandasColumn.get_chunks` (pandas/core/interchange/column.py:250-265). Both implementations share the same flawed logic:

```python
for start in range(0, step * n_chunks, step):  # Bug: iterates beyond size
    yield PandasDataFrameXchg(
        self._df.iloc[start : start + step, :],
        allow_copy=self._allow_copy,
    )
```

When `size % n_chunks != 0`, the step is incremented by 1, causing `step * n_chunks` to exceed the DataFrame's actual size. For example, with size=11 and n_chunks=5:
- step = 11 // 5 = 2
- Since 11 % 5 != 0, step becomes 3
- Loop iterates: range(0, 15, 3) = [0, 3, 6, 9, 12]
- The slice [12:15] on an 11-row DataFrame returns an empty chunk

The dataframe interchange protocol specification doesn't explicitly forbid empty chunks, but the practical purpose of chunking (parallel processing, load balancing) makes empty chunks counterproductive. Other implementations like PyArrow include checks to avoid empty batches.

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

--- a/pandas/core/interchange/column.py
+++ b/pandas/core/interchange/column.py
@@ -257,7 +257,7 @@ class PandasColumn(Column):
             step = size // n_chunks
             if size % n_chunks != 0:
                 step += 1
-            for start in range(0, step * n_chunks, step):
+            for start in range(0, size, step):
                 yield PandasColumn(
                     self._col.iloc[start : start + step], self._allow_copy
                 )
```