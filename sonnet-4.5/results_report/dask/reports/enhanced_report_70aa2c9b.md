# Bug Report: dask.dataframe.dask_expr.io.bag.to_bag format='frame' Returns Column Names Instead of DataFrames

**Target**: `dask.dataframe.dask_expr.io.bag.to_bag`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `to_bag` function with `format='frame'` returns a Bag containing column names (strings) instead of DataFrame partitions, violating its documented behavior which states that "the original partitions of df will not be transformed in any way."

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume, settings
from hypothesis.extra.pandas import data_frames, columns
import pandas as pd
from dask.dataframe.dask_expr import from_pandas
from dask.dataframe.dask_expr.io.bag import to_bag
import dask

# Use synchronous scheduler to avoid multiprocessing issues
dask.config.set(scheduler='synchronous')

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

    assert len(result) == npartitions, f"Expected {npartitions} items, got {len(result)}"
    assert all(isinstance(item, pd.DataFrame) for item in result), f"Expected all items to be DataFrames, got types: {[type(item) for item in result]}"

if __name__ == '__main__':
    test_to_bag_frame_format_should_preserve_dataframes()
```

<details>

<summary>
**Failing input**: `df=pd.DataFrame({'A': [1.0], 'B': [2.0]}), npartitions=1`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/63/hypo.py", line 30, in <module>
    test_to_bag_frame_format_should_preserve_dataframes()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/63/hypo.py", line 12, in test_to_bag_frame_format_should_preserve_dataframes
    df=data_frames(
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/63/hypo.py", line 26, in test_to_bag_frame_format_should_preserve_dataframes
    assert len(result) == npartitions, f"Expected {npartitions} items, got {len(result)}"
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Expected 1 items, got 2
Falsifying example: test_to_bag_frame_format_should_preserve_dataframes(
    df=
             A    B
        0  1.0  2.0
    ,
    npartitions=1,
)
```
</details>

## Reproducing the Bug

```python
import pandas as pd
from dask.dataframe.dask_expr import from_pandas
from dask.dataframe.dask_expr.io.bag import to_bag
import dask

# Use synchronous scheduler to avoid multiprocessing issues
dask.config.set(scheduler='synchronous')

if __name__ == '__main__':
    # Create a simple DataFrame
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    print("Original DataFrame:")
    print(df)
    print()

    # Convert to Dask DataFrame with 2 partitions
    ddf = from_pandas(df, npartitions=2)
    print(f"Dask DataFrame with {ddf.npartitions} partitions")
    print()

    # Try to convert to bag with format='frame'
    print("Attempting to_bag(ddf, format='frame')...")
    bag = to_bag(ddf, format='frame')
    result = bag.compute()

    print(f"Expected: {ddf.npartitions} DataFrame objects")
    print(f"Got: {len(result)} items of type {type(result[0]) if result else 'None'}")
    print(f"Result: {result}")
    print()

    # Show what each partition should look like
    print("What we expected (DataFrame partitions):")
    for i in range(ddf.npartitions):
        partition = ddf.get_partition(i).compute()
        print(f"Partition {i}:")
        print(partition)
        print()
```

<details>

<summary>
AssertionError: Got 4 string items instead of 2 DataFrames
</summary>
```
Original DataFrame:
   A  B
0  1  4
1  2  5
2  3  6

Dask DataFrame with 2 partitions

Attempting to_bag(ddf, format='frame')...
Expected: 2 DataFrame objects
Got: 4 items of type <class 'str'>
Result: ['A', 'B', 'A', 'B']

What we expected (DataFrame partitions):
Partition 0:
   A  B
0  1  4
1  2  5

Partition 1:
   A  B
2  3  6

```
</details>

## Why This Is A Bug

The docstring for `to_bag` explicitly states that when `format='frame'` is specified, it should return "dataframe-like objects" and that "the original partitions of df will not be transformed in any way." However, the implementation returns column names as strings instead of preserving the DataFrame partitions.

This occurs because when Python iterates over a pandas DataFrame, it yields the column names rather than rows or the DataFrame itself. The current implementation at lines 34-36 of `bag.py` directly assigns `df.dask` to the Bag's task graph without wrapping each DataFrame partition. When `Bag.compute()` collects the results, it iterates over each DataFrame partition, yielding column names instead of preserving the DataFrame as a single object.

The other formats (`'tuple'` and `'dict'`) work correctly because they use the `_df_to_bag` helper function which properly transforms DataFrames into the desired format. The `format='frame'` case bypasses this helper entirely but fails to account for DataFrame iteration behavior.

## Relevant Context

- The `_df_to_bag` helper function in `/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages/dask/dataframe/io/io.py` (lines 237-254) only handles `'tuple'` and `'dict'` formats, not `'frame'`
- The bug affects all DataFrame sizes and partition counts - even a single partition returns column names instead of the DataFrame
- Users can work around this by using `ddf.get_partition(i).compute()` to manually extract partitions
- The issue is in the core logic at `/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages/dask/dataframe/dask_expr/io/bag.py` lines 34-36

## Proposed Fix

```diff
--- a/dask/dataframe/dask_expr/io/bag.py
+++ b/dask/dataframe/dask_expr/io/bag.py
@@ -32,8 +32,11 @@ def to_bag(df, index=False, format="tuple"):
         raise TypeError("df must be either DataFrame or Series")
     name = "to_bag-" + tokenize(df._name, index, format)
     if format == "frame":
-        dsk = df.dask
-        name = df._name
+        # Wrap each partition in a list to prevent iteration over DataFrames
+        dsk = {
+            (name, i): (lambda x: [x], block)
+            for (i, block) in enumerate(df.__dask_keys__())
+        }
+        dsk.update(df.__dask_graph__())
     else:
         dsk = {
             (name, i): (_df_to_bag, block, index, format)
```