# Bug Report: dask.dataframe.dask_expr.io.bag.to_bag format='frame' Returns Column Names Instead of DataFrames

**Target**: `dask.dataframe.dask_expr.io.bag.to_bag`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `to_bag` function with `format='frame'` returns a Bag containing column names (strings) instead of DataFrame partitions. According to the docstring, `format='frame'` should return "dataframe-like objects" where "the original partitions of df will not be transformed in any way", but instead it iterates over the DataFrame partitions yielding column names.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume, settings
from hypothesis.extra.pandas import data_frames, columns
import pandas as pd
from dask.dataframe.dask_expr import from_pandas
from dask.dataframe.dask_expr.io.bag import to_bag


@given(
    df=data_frames(
        columns=columns(['A', 'B'], dtype=float),
        rows=st.tuples(st.just(1), st.integers(min_value=2, max_value=10))
    ),
    npartitions=st.integers(min_value=1, max_value=3)
)
@settings(max_examples=50, deadline=None)
def test_to_bag_frame_format_should_preserve_dataframes(df, npartitions):
    assume(len(df) >= npartitions)

    ddf = from_pandas(df, npartitions=npartitions)
    bag = to_bag(ddf, format='frame', index=False)
    result = bag.compute()

    assert len(result) == npartitions
    assert all(isinstance(item, pd.DataFrame) for item in result)
```

**Failing input**: Any DataFrame with 2+ partitions, e.g., `pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})` with `npartitions=2`

## Reproducing the Bug

```python
import pandas as pd
from dask.dataframe.dask_expr import from_pandas
from dask.dataframe.dask_expr.io.bag import to_bag

df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
ddf = from_pandas(df, npartitions=2)

bag = to_bag(ddf, format='frame')
result = bag.compute()

print(f"Expected: {ddf.npartitions} DataFrame objects")
print(f"Got: {len(result)} items")
print(f"Result: {result}")
```

**Output:**
```
Expected: 2 DataFrame objects
Got: 4 items
Result: ['A', 'B', 'A', 'B']
```

## Why This Is A Bug

The docstring states that `format='frame'` should return a bag of "dataframe-like objects" and that "the original partitions of df will not be transformed in any way". Users would expect each bag element to be a DataFrame (one per partition), but instead the Bag iterates over the DataFrames, yielding column names.

This violates the documented behavior and makes `format='frame'` unusable for its intended purpose.

## Fix

The issue is in lines 34-36 of `bag.py`:

```python
if format == "frame":
    dsk = df.dask
    name = df._name
```

This creates a Bag directly from the DataFrame's task graph, but `Bag.compute()` iterates over each result. Since DataFrames are iterable (yielding column names), we get strings instead of DataFrames.

The fix should ensure each DataFrame partition is treated as a single Bag element:

```diff
--- a/dask/dataframe/dask_expr/io/bag.py
+++ b/dask/dataframe/dask_expr/io/bag.py
@@ -32,7 +32,11 @@ def to_bag(df, index=False, format="tuple"):
         raise TypeError("df must be either DataFrame or Series")
     name = "to_bag-" + tokenize(df._name, index, format)
     if format == "frame":
-        dsk = df.dask
-        name = df._name
+        # Wrap each partition so it's not iterated over
+        dsk = {
+            (name, i): (lambda x: [x], block)
+            for (i, block) in enumerate(df.__dask_keys__())
+        }
+        dsk.update(df.__dask_graph__())
     else:
         dsk = {
```

This wraps each DataFrame partition in a list before adding it to the Bag, preventing unwanted iteration.