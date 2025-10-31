# Bug Report: dask.dataframe.io.orc Empty DataFrame Read Failure

**Target**: `dask.dataframe.io.orc.read_orc`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `read_orc` function crashes with a ValueError when attempting to read ORC files that were successfully written from empty DataFrames, violating the fundamental round-trip property of I/O operations.

## Property-Based Test

```python
import tempfile
import shutil
import pandas as pd
import dask.dataframe as dd
from hypothesis import given, strategies as st, settings


@settings(max_examples=50, deadline=10000)
@given(
    num_rows=st.integers(min_value=0, max_value=50),
)
def test_orc_empty_dataframe_round_trip(num_rows):
    df_pandas = pd.DataFrame({
        'col_a': list(range(num_rows)),
        'col_b': [f'val_{i}' for i in range(num_rows)],
    })
    df = dd.from_pandas(df_pandas, npartitions=1)

    tmpdir = tempfile.mkdtemp()
    try:
        dd.to_orc(df, tmpdir, write_index=False)
        df_read = dd.read_orc(tmpdir)
        df_result = df_read.compute()

        assert len(df_result) == num_rows, \
            f"Row count mismatch: {len(df_result)} != {num_rows}"

        if num_rows == 0:
            assert list(df_result.columns) == ['col_a', 'col_b'], \
                "Column names should be preserved even for empty dataframes"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
```

<details>

<summary>
**Failing input**: `num_rows=0`
</summary>
```
Falsifying example: test_orc_empty_dataframe_round_trip(
    num_rows=0,
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/miniconda/lib/python3.13/site-packages/dask/backends.py:145
        /home/npc/miniconda/lib/python3.13/site-packages/dask/backends.py:146
        /home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/dask_expr/_collection.py:5895
        /home/npc/miniconda/lib/python3.13/site-packages/pandas/core/construction.py:645
        /home/npc/miniconda/lib/python3.13/site-packages/pandas/core/construction.py:816
        (and 2 more with settings.verbosity >= verbose)
Traceback (most recent call last):
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/backends.py", line 140, in wrapper
    return func(*args, **kwargs)
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/io/orc/core.py", line 99, in read_orc
    return dd.from_map(
           ~~~~~~~~~~~^
        _read_orc,
        ^^^^^^^^^^
    ...<6 lines>...
        columns=columns,
        ^^^^^^^^^^^^^^^^
    )
    ^
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/dask_expr/_collection.py", line 5895, in from_map
    raise ValueError("All `iterables` must have a non-zero length")
ValueError: All `iterables` must have a non-zero length

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/19/hypo.py", line 37, in <module>
    test_orc_empty_dataframe_round_trip()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/19/hypo.py", line 9, in test_orc_empty_dataframe_round_trip
    @given(

  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/19/hypo.py", line 22, in test_orc_empty_dataframe_round_trip
    df_read = dd.read_orc(tmpdir)
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/backends.py", line 151, in wrapper
    raise exc from e
ValueError: An error occurred while calling the read_orc method registered to the pandas backend.
Original Message: All `iterables` must have a non-zero length
```
</details>

## Reproducing the Bug

```python
import tempfile
import shutil
import pandas as pd
import dask.dataframe as dd

tmpdir = tempfile.mkdtemp()
try:
    # Create an empty pandas DataFrame with columns but no rows
    df_pandas = pd.DataFrame({'col_a': [], 'col_b': []})
    print(f"Created empty DataFrame: shape={df_pandas.shape}, columns={list(df_pandas.columns)}")

    # Convert to Dask DataFrame
    df = dd.from_pandas(df_pandas, npartitions=1)
    print(f"Converted to Dask DataFrame: npartitions={df.npartitions}")

    # Write to ORC format
    dd.to_orc(df, tmpdir, write_index=False)
    print(f"Successfully wrote empty DataFrame to ORC in {tmpdir}")

    # List the ORC files created
    import os
    files = os.listdir(tmpdir)
    print(f"ORC files created: {files}")

    # Try to read the ORC file back
    print("Attempting to read ORC file...")
    df_read = dd.read_orc(tmpdir)
    print("Successfully read ORC file")

    df_result = df_read.compute()
    print(f"Result DataFrame: shape={df_result.shape}, columns={list(df_result.columns)}")

except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
    import traceback
    print("\nFull traceback:")
    traceback.print_exc()
finally:
    shutil.rmtree(tmpdir, ignore_errors=True)
```

<details>

<summary>
ERROR: ValueError when reading empty ORC file
</summary>
```
Created empty DataFrame: shape=(0, 2), columns=['col_a', 'col_b']
Converted to Dask DataFrame: npartitions=1
Successfully wrote empty DataFrame to ORC in /tmp/tmpyx7zbi08
ORC files created: ['part.0.orc']
Attempting to read ORC file...
ERROR: ValueError: An error occurred while calling the read_orc method registered to the pandas backend.
Original Message: All `iterables` must have a non-zero length

Full traceback:
Traceback (most recent call last):
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/backends.py", line 140, in wrapper
    return func(*args, **kwargs)
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/io/orc/core.py", line 99, in read_orc
    return dd.from_map(
           ~~~~~~~~~~~^
        _read_orc,
        ^^^^^^^^^^
    ...<6 lines>...
        columns=columns,
        ^^^^^^^^^^^^^^^^
    )
    ^
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/dask_expr/_collection.py", line 5895, in from_map
    raise ValueError("All `iterables` must have a non-zero length")
ValueError: All `iterables` must have a non-zero length

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/19/repo.py", line 27, in <module>
    df_read = dd.read_orc(tmpdir)
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/backends.py", line 151, in wrapper
    raise exc from e
ValueError: An error occurred while calling the read_orc method registered to the pandas backend.
Original Message: All `iterables` must have a non-zero length
```
</details>

## Why This Is A Bug

This bug violates the fundamental round-trip property expected in data I/O operations. When `dd.to_orc()` successfully writes an empty DataFrame to disk without any errors or warnings, users reasonably expect `dd.read_orc()` to be able to read that file back. The behavior contradicts several important principles:

1. **Valid Data Structure**: Empty DataFrames are legitimate objects in both pandas and Dask, commonly resulting from filtering operations that match no rows, initialization scenarios, or time series data with gaps.

2. **Format Support**: The ORC format itself has no issues with empty files. PyArrow (the underlying engine used by Dask) can correctly read these empty ORC files, proving they are valid ORC files with proper metadata but zero data stripes.

3. **Inconsistent Behavior**: The write operation completes successfully without any warnings, creating a valid ORC file that preserves the schema information. However, the read operation fails due to an implementation detail in `dd.from_map()` that requires non-empty iterables.

4. **Documentation Gap**: The documentation for both `read_orc` and `to_orc` does not mention any limitations regarding empty DataFrames, nor does it warn users that files written by `to_orc` might not be readable by `read_orc`.

## Relevant Context

The bug occurs in `/home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/io/orc/core.py:99` when `read_orc` calls `dd.from_map()` with an empty `parts` list. When an ORC file has no data stripes (as is the case with empty DataFrames), the `ArrowORCEngine.read_metadata()` method returns an empty list for `parts`, which causes `dd.from_map()` to raise a ValueError.

Key code locations:
- Bug location: `dask/dataframe/io/orc/core.py:99`
- Error source: `dask/dataframe/dask_expr/_collection.py:5895`
- Metadata extraction: `dask/dataframe/io/orc/arrow.py:88` (lines 36-63 handle stripe extraction)

The ORC file created by `to_orc` is valid and contains:
- Proper ORC file structure with headers and footers
- Schema information with column names and types
- Zero stripes (data sections) since there are no rows

## Proposed Fix

```diff
--- a/dask/dataframe/io/orc/core.py
+++ b/dask/dataframe/io/orc/core.py
@@ -96,6 +96,11 @@ def read_orc(

     if columns is not None and index in columns:
         columns = [col for col in columns if col != index]
+
+    # Handle empty ORC files (no stripes/partitions)
+    if not parts:
+        # Return a single-partition DataFrame with the correct schema but no data
+        return dd.from_pandas(meta, npartitions=1)
+
     return dd.from_map(
         _read_orc,
         parts,
```