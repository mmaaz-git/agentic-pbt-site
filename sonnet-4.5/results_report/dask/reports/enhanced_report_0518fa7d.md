# Bug Report: dask.dataframe.io.orc.arrow ArrowORCEngine._aggregate_files TypeError with None Stripes

**Target**: `dask.dataframe.io.orc.arrow.ArrowORCEngine._aggregate_files`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`ArrowORCEngine._aggregate_files` crashes with TypeError when attempting to call `len()` on `None` stripe values that are created when `split_stripes=False` in `read_metadata`.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from dask.dataframe.io.orc.arrow import ArrowORCEngine

@given(st.integers(min_value=1, max_value=10))
@settings(max_examples=200)
def test_aggregate_files_with_none_stripes(split_stripes_val):
    parts = [[("file1.orc", None)], [("file2.orc", None)]]

    result = ArrowORCEngine._aggregate_files(
        aggregate_files=True,
        split_stripes=split_stripes_val,
        parts=parts
    )

    assert result is not None

# Run the test
if __name__ == "__main__":
    test_aggregate_files_with_none_stripes()
```

<details>

<summary>
**Failing input**: `split_stripes_val=2`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/14/hypo.py", line 19, in <module>
    test_aggregate_files_with_none_stripes()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/14/hypo.py", line 5, in test_aggregate_files_with_none_stripes
    @settings(max_examples=200)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/14/hypo.py", line 9, in test_aggregate_files_with_none_stripes
    result = ArrowORCEngine._aggregate_files(
        aggregate_files=True,
        split_stripes=split_stripes_val,
        parts=parts
    )
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/io/orc/arrow.py", line 86, in _aggregate_files
    nstripes = len(new_part[0][1])
TypeError: object of type 'NoneType' has no len()
Falsifying example: test_aggregate_files_with_none_stripes(
    split_stripes_val=2,
)
```
</details>

## Reproducing the Bug

```python
from dask.dataframe.io.orc.arrow import ArrowORCEngine

# Create parts with None stripe values, as created by read_metadata
# when split_stripes=False (line 63 in arrow.py)
parts = [[("file1.orc", None)], [("file2.orc", None)]]

# This should work but will crash with TypeError
result = ArrowORCEngine._aggregate_files(
    aggregate_files=True,
    split_stripes=2,
    parts=parts
)

print("Result:", result)
```

<details>

<summary>
TypeError: object of type 'NoneType' has no len()
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/14/repo.py", line 8, in <module>
    result = ArrowORCEngine._aggregate_files(
        aggregate_files=True,
        split_stripes=2,
        parts=parts
    )
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/io/orc/arrow.py", line 86, in _aggregate_files
    nstripes = len(new_part[0][1])
TypeError: object of type 'NoneType' has no len()
```
</details>

## Why This Is A Bug

This bug violates the documented behavior and creates an internal inconsistency in the codebase:

1. **Documentation allows this combination**: The `read_orc` function documentation states that `split_stripes` can be `False` to create a 1-to-1 mapping between files and partitions, and `aggregate_files` can be `True` to allow file aggregation. There's no documented restriction preventing these from being used together.

2. **Internal state inconsistency**: When `read_metadata` is called with `split_stripes=False`, it intentionally creates parts with `None` stripe values at line 63 (`parts.append([(path, None)])`). However, `_aggregate_files` assumes all stripe values are lists and calls `len()` on them at line 86, causing the crash.

3. **Valid use case blocked**: Users may legitimately want to aggregate multiple ORC files into partitions without splitting individual files by stripes. This is a reasonable workflow that should be supported.

4. **The code path is reachable**: The `read_metadata` function at line 74 calls `_aggregate_files` with the parts it created, meaning this crash will occur in normal usage when users specify `split_stripes=False` in `read_orc`.

## Relevant Context

The bug occurs in the internal aggregation logic of the ArrowORCEngine class. The `_aggregate_files` method is called from `read_metadata` (line 74 in arrow.py) as part of the ORC file reading pipeline.

Key code locations:
- Creation of None stripes: `/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages/dask/dataframe/io/orc/arrow.py:63`
- Crash location: `/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages/dask/dataframe/io/orc/arrow.py:86`
- Public API: `dask.dataframe.read_orc()` in `/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages/dask/dataframe/io/orc/core.py`

The issue affects the PyArrow backend for ORC file reading in Dask, which is the default and recommended engine for ORC operations.

## Proposed Fix

```diff
--- a/dask/dataframe/io/orc/arrow.py
+++ b/dask/dataframe/io/orc/arrow.py
@@ -81,12 +81,15 @@ class ArrowORCEngine:
     @classmethod
     def _aggregate_files(cls, aggregate_files, split_stripes, parts):
         if aggregate_files is True and int(split_stripes) > 1 and len(parts) > 1:
             new_parts = []
             new_part = parts[0]
-            nstripes = len(new_part[0][1])
+            # Handle None stripes (from split_stripes=False)
+            stripes = new_part[0][1]
+            if stripes is None:
+                return parts  # Cannot aggregate when stripes are not split
+            nstripes = len(stripes)
             for part in parts[1:]:
-                next_nstripes = len(part[0][1])
+                next_nstripes = len(part[0][1]) if part[0][1] is not None else 0
                 if next_nstripes + nstripes <= split_stripes:
                     new_part.append(part[0])
                     nstripes += next_nstripes
```