# Bug Report: pandas.core.interchange.dataframe.PandasDataFrameXchg.get_chunks Creates Empty Chunks

**Target**: `pandas.core.interchange.dataframe.PandasDataFrameXchg.get_chunks`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When `n_chunks` exceeds the number of rows in a DataFrame, `get_chunks()` creates empty chunks instead of limiting chunks to available data.

## Property-Based Test

```python
import pandas as pd
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


if __name__ == "__main__":
    test_chunk_sizes_reasonable()
```

<details>

<summary>
**Failing input**: `nrows=1, n_chunks=2`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/19/hypo.py", line 21, in <module>
    test_chunk_sizes_reasonable()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/19/hypo.py", line 6, in test_chunk_sizes_reasonable
    st.integers(min_value=1, max_value=100),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/19/hypo.py", line 17, in test_chunk_sizes_reasonable
    assert size > 0, f"Chunk has {size} rows (should be > 0)"
           ^^^^^^^^
AssertionError: Chunk has 0 rows (should be > 0)
Falsifying example: test_chunk_sizes_reasonable(
    nrows=1,
    n_chunks=2,  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
import pandas as pd

df = pd.DataFrame({'a': [0]})
interchange_df = df.__dataframe__()

chunks = list(interchange_df.get_chunks(n_chunks=2))

print(f"DataFrame has {df.shape[0]} rows")
print(f"Requested {2} chunks")
print(f"Got {len(chunks)} chunks")
print(f"Chunk 0: {chunks[0].num_rows()} rows")
print(f"Chunk 1: {chunks[1].num_rows()} rows")
```

<details>

<summary>
Output showing empty chunk creation
</summary>
```
DataFrame has 1 rows
Requested 2 chunks
Got 2 chunks
Chunk 0: 1 rows
Chunk 1: 0 rows
```
</details>

## Why This Is A Bug

This violates expected behavior of the interchange protocol in several ways:

1. **Useless empty chunks**: Empty chunks serve no purpose in data interchange. The protocol is designed to transfer data, and empty containers don't contribute to that goal.

2. **Unexpected behavior**: Users reasonably expect that chunks will contain data. The specification in `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/core/interchange/dataframe_protocol.py` states that `get_chunks()` should "Return an iterator yielding the chunks", implying chunks contain actual data.

3. **Inconsistent with protocol intent**: While the specification doesn't explicitly prohibit empty chunks, it states that if `n_chunks` is given, "the producer must subdivide each chunk", not create new empty ones.

4. **Potential consumer failures**: Libraries consuming the interchange protocol may not handle empty chunks correctly, potentially causing downstream errors or unexpected behavior.

## Relevant Context

The bug occurs in the `get_chunks` method at lines 102-111 of `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/core/interchange/dataframe.py`. The problematic logic:

1. When `n_chunks=2` and DataFrame has 1 row:
   - `step = 1 // 2 = 0`, then incremented to `1`
   - `range(0, step * n_chunks, step)` = `range(0, 2, 1)` = `[0, 1]`
   - This creates 2 iterations, but the second starts at row index 1, which doesn't exist
   - Result: Second chunk is empty with 0 rows

The issue is that the loop iterates `n_chunks` times regardless of available data. The range should be limited to the actual size of the DataFrame.

Documentation reference: The interchange protocol is defined in [pandas interchange protocol](https://pandas.pydata.org/docs/reference/api/pandas.api.interchange.from_dataframe.html) and the specification is vendored from [data-apis/dataframe-api](https://github.com/data-apis/dataframe-api).

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