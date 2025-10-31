# Bug Report: dask.dataframe.io.orc Index Duplication in Multi-Partition Round-Trip

**Target**: `dask.dataframe.io.orc.read_orc` and `dask.dataframe.io.orc.to_orc`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When writing a multi-partition Dask DataFrame to ORC format without the index (`write_index=False`) and reading it back, the resulting DataFrame contains duplicate index values instead of a sequential RangeIndex, causing silent data corruption.

## Property-Based Test

```python
import tempfile
import pandas as pd
import dask.dataframe as dd
from hypothesis import given, strategies as st, settings

@given(
    st.lists(st.integers(), min_size=1, max_size=100),
    st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=1, max_size=100),
)
@settings(max_examples=50)
def test_round_trip_without_index(int_data, float_data):
    min_len = min(len(int_data), len(float_data))
    df_pandas = pd.DataFrame({
        'x': int_data[:min_len],
        'y': float_data[:min_len],
    })
    df = dd.from_pandas(df_pandas, npartitions=2)

    with tempfile.TemporaryDirectory() as tmpdir:
        dd.io.orc.to_orc(df, tmpdir, write_index=False)
        df_loaded = dd.io.orc.read_orc(tmpdir)
        df_result = df_loaded.compute()
        df_expected = df.compute().reset_index(drop=True)

        pd.testing.assert_frame_equal(df_result, df_expected)

if __name__ == "__main__":
    test_round_trip_without_index()
```

<details>

<summary>
**Failing input**: `int_data=[0, 0], float_data=[0.0, 0.0]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/37/hypo.py", line 28, in <module>
    test_round_trip_without_index()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/37/hypo.py", line 7, in test_round_trip_without_index
    st.lists(st.integers(), min_size=1, max_size=100),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/37/hypo.py", line 25, in test_round_trip_without_index
    pd.testing.assert_frame_equal(df_result, df_expected)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/_testing/asserters.py", line 1250, in assert_frame_equal
    assert_index_equal(
    ~~~~~~~~~~~~~~~~~~^
        left.index,
        ^^^^^^^^^^^
    ...<8 lines>...
        obj=f"{obj}.index",
        ^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/_testing/asserters.py", line 326, in assert_index_equal
    _testing.assert_almost_equal(
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        left.values,
        ^^^^^^^^^^^^
    ...<6 lines>...
        robj=right,
        ^^^^^^^^^^^
    )
    ^
  File "pandas/_libs/testing.pyx", line 55, in pandas._libs.testing.assert_almost_equal
  File "pandas/_libs/testing.pyx", line 173, in pandas._libs.testing.assert_almost_equal
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/_testing/asserters.py", line 620, in raise_assert_detail
    raise AssertionError(msg)
AssertionError: DataFrame.index are different

DataFrame.index values are different (50.0 %)
[left]:  Index([0, 0], dtype='int64')
[right]: RangeIndex(start=0, stop=2, step=1)
At positional index 1, first diff: 0 != 1
Falsifying example: test_round_trip_without_index(
    int_data=[0, 0],
    float_data=[0.0, 0.0],
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/miniconda/lib/python3.13/site-packages/dask/_task_spec.py:665
        /home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/backends.py:746
        /home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/backends.py:747
        /home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/backends.py:748
        /home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/backends.py:750
        (and 115 more with settings.verbosity >= verbose)
```
</details>

## Reproducing the Bug

```python
import tempfile
import pandas as pd
import dask.dataframe as dd

# Create a minimal test case
df_pandas = pd.DataFrame({
    'x': [0, 0],
    'y': [0.0, 0.0],
})

# Convert to Dask DataFrame with 2 partitions
df = dd.from_pandas(df_pandas, npartitions=2)

with tempfile.TemporaryDirectory() as tmpdir:
    # Write to ORC without index
    dd.io.orc.to_orc(df, tmpdir, write_index=False)

    # Read back from ORC
    df_loaded = dd.io.orc.read_orc(tmpdir)
    df_result = df_loaded.compute()

    print("Original DataFrame:")
    print("Index:", list(df.compute().index))
    print("Values:\n", df.compute())
    print("\nLoaded DataFrame:")
    print("Index:", list(df_result.index))
    print("Values:\n", df_result)
    print("\nExpected index: [0, 1]")
    print("Actual index:  ", list(df_result.index))
    print("\nBUG: The indices are duplicated! Both values are 0 instead of [0, 1]")
```

<details>

<summary>
Output showing duplicate indices
</summary>
```
Original DataFrame:
Index: [0, 1]
Values:
    x    y
0  0  0.0
1  0  0.0

Loaded DataFrame:
Index: [0, 0]
Values:
    x    y
0  0  0.0
0  0  0.0

Expected index: [0, 1]
Actual index:   [0, 0]

BUG: The indices are duplicated! Both values are 0 instead of [0, 1]
```
</details>

## Why This Is A Bug

This violates expected DataFrame behavior and causes data corruption for the following reasons:

1. **Duplicate indices break DataFrame semantics**: DataFrames should have unique indices for proper data alignment and operations. The duplicate index values `[0, 0]` instead of `[0, 1]` violate this fundamental constraint.

2. **Silent data corruption**: The round-trip operation silently corrupts the data structure. Users expect that writing and reading data back should preserve the logical structure, even if the index isn't explicitly stored.

3. **Undocumented behavior**: Neither the `to_orc` nor `read_orc` documentation mentions that using `write_index=False` will result in duplicate indices when reading multi-partition data back. The `to_orc` documentation only states "Whether or not to write the index" without warning about this consequence.

4. **Root cause**: When `write_index=False` is used, the `to_orc` function resets the index globally (line 180-181 in core.py: `df = df.reset_index(drop=True)`). However, when each partition is written to a separate ORC file (as documented: "Each partition will be written to a separate file"), each file contains data with a local RangeIndex starting from 0. When `read_orc` reads these files back, it preserves these per-partition local indices instead of reconstructing a global sequential index, resulting in duplicates.

5. **Inconsistent with user expectations**: Users reasonably expect that a DataFrame round-trip operation should either preserve indices or recreate valid, non-duplicate indices. The current behavior does neither.

## Relevant Context

- This issue affects multi-partition DataFrames when using `write_index=False` (non-default option)
- The same bug exists in other Dask I/O formats (Parquet, CSV), suggesting it's a systematic issue in how Dask handles index-less round-trips
- Each partition file (part.0.orc, part.1.orc, etc.) contains correct data but with local indices starting from 0
- The bug occurs in the read phase when these partition-local indices are not adjusted to be globally unique
- Source code locations:
  - `/dask/dataframe/io/orc/core.py:180-181` - Index reset during write
  - `/dask/dataframe/io/orc/core.py:99-108` - Read operation that preserves local indices

## Proposed Fix

The issue can be fixed by resetting the index to a sequential RangeIndex after reading when no explicit index column is specified:

```diff
--- a/dask/dataframe/io/orc/core.py
+++ b/dask/dataframe/io/orc/core.py
@@ -96,11 +96,16 @@ def read_orc(

     if columns is not None and index in columns:
         columns = [col for col in columns if col != index]
-    return dd.from_map(
+
+    result = dd.from_map(
         _read_orc,
         parts,
         engine=engine,
         fs=fs,
         schema=schema,
         index=index,
         meta=meta,
         columns=columns,
     )
+
+    # Reset index to create a global sequential RangeIndex when no explicit index is specified
+    if index is None:
+        result = result.reset_index(drop=True)
+
+    return result
```