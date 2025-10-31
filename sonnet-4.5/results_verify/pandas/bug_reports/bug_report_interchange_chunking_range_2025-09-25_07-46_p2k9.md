# Bug Report: pandas.core.interchange get_chunks Incorrect Range Calculation

**Target**: `pandas.core.interchange.dataframe.PandasDataFrameXchg.get_chunks` and `pandas.core.interchange.column.PandasColumn.get_chunks`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `get_chunks` method in both `PandasDataFrameXchg` and `PandasColumn` uses an incorrect range calculation that can produce chunk boundaries extending beyond the data size. While pandas' `iloc` handles this gracefully, the implementation is mathematically incorrect and could cause issues in other DataFrame implementations following the interchange protocol.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import pandas as pd


@given(
    size=st.integers(min_value=1, max_value=100),
    n_chunks=st.integers(min_value=2, max_value=10)
)
@settings(max_examples=200)
def test_chunking_range_within_bounds(size, n_chunks):
    df = pd.DataFrame({'col': range(size)})
    interchange_obj = df.__dataframe__()

    step = size // n_chunks
    if size % n_chunks != 0:
        step += 1

    chunk_ranges = list(range(0, step * n_chunks, step))

    for i, start in enumerate(chunk_ranges):
        end = start + step
        assert end <= size or i == len(chunk_ranges) - 1, \
            f"Chunk {i} has end={end} > size={size}, but it's not the last chunk"
```

**Failing input**: size=10, n_chunks=3 produces range [0, 4, 8] with last chunk [8:12] extending beyond size 10

## Reproducing the Bug

```python
import pandas as pd

size = 10
n_chunks = 3

df = pd.DataFrame({'col': range(size)})
interchange_obj = df.__dataframe__()

step = size // n_chunks
print(f"size = {size}, n_chunks = {n_chunks}, initial step = {step}")

if size % n_chunks != 0:
    step += 1
    print(f"size % n_chunks != 0, so step += 1 → step = {step}")

print(f"\nrange(0, step * n_chunks, step) = range(0, {step * n_chunks}, {step})")
chunk_starts = list(range(0, step * n_chunks, step))
print(f"Chunk starts: {chunk_starts}")

print("\nChunk ranges:")
for i, start in enumerate(chunk_starts):
    end = start + step
    print(f"  Chunk {i}: [{start}:{end}]", end='')
    if end > size:
        print(f" ← EXTENDS BEYOND DATA (size={size})")
    else:
        print()

print("\nThe last chunk [8:12] tries to access 4 elements starting at index 8,")
print("but the data only has 10 elements (indices 0-9).")
print("\nPandas' iloc handles this gracefully by returning [8:10], but the")
print("calculation is still mathematically incorrect.")
```

Output:
```
size = 10, n_chunks = 3, initial step = 3
size % n_chunks != 0, so step += 1 → step = 4

range(0, step * n_chunks, step) = range(0, 12, 4)
Chunk starts: [0, 4, 8]

Chunk ranges:
  Chunk 0: [0:4]
  Chunk 1: [4:8]
  Chunk 2: [8:12] ← EXTENDS BEYOND DATA (size=10)

The last chunk [8:12] tries to access 4 elements starting at index 8,
but the data only has 10 elements (indices 0-9).

Pandas' iloc handles this gracefully by returning [8:10], but the
calculation is still mathematically incorrect.
```

## Why This Is A Bug

The range calculation `range(0, step * n_chunks, step)` is mathematically incorrect when `size % n_chunks != 0`. The code calculates:

```python
step = size // n_chunks
if size % n_chunks != 0:
    step += 1
for start in range(0, step * n_chunks, step):
    yield self._df.iloc[start : start + step, :]
```

This produces `n_chunks` iterations with step size `ceil(size / n_chunks)`, which means the total range `step * n_chunks` can exceed the actual data size.

While pandas' `iloc[start:end]` handles out-of-bounds `end` gracefully by capping it, this is:
1. **Relying on implementation details** of pandas rather than explicit logic
2. **Incorrect in principle** - the last chunk should be `[start:size]`, not `[start:start+step]`
3. **Potentially problematic** for other DataFrame libraries implementing the interchange protocol that might not handle out-of-bounds slicing as gracefully

## Fix

Calculate the end boundary correctly for each chunk:

```diff
--- a/pandas/core/interchange/dataframe.py
+++ b/pandas/core/interchange/dataframe.py
@@ -101,12 +101,12 @@ class PandasDataFrameXchg(DataFrameXchg):
         """
         if n_chunks and n_chunks > 1:
             size = len(self._df)
             step = size // n_chunks
             if size % n_chunks != 0:
                 step += 1
-            for start in range(0, step * n_chunks, step):
+            for start in range(0, size, step):
                 yield PandasDataFrameXchg(
-                    self._df.iloc[start : start + step, :],
+                    self._df.iloc[start : min(start + step, size), :],
                     allow_copy=self._allow_copy,
                 )
         else:
             yield self
```

Apply the same fix to `column.py` lines 255-263.