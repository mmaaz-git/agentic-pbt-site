# Bug Report: dask.dataframe.io.orc ArrowORCEngine._aggregate_files TypeError with None Stripes

**Target**: `dask.dataframe.io.orc.arrow.ArrowORCEngine._aggregate_files`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`ArrowORCEngine._aggregate_files` crashes with TypeError when `aggregate_files=True`, `split_stripes > 1`, and parts contain `None` stripe values (which occurs when `split_stripes=False` in `read_metadata`).

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
```

**Failing input**: `split_stripes_val=2`

## Reproducing the Bug

```python
from dask.dataframe.io.orc.arrow import ArrowORCEngine

parts = [[("file1.orc", None)], [("file2.orc", None)]]

result = ArrowORCEngine._aggregate_files(
    aggregate_files=True,
    split_stripes=2,
    parts=parts
)
```

## Why This Is A Bug

When `read_metadata` is called with `split_stripes=False`, it creates parts with `None` stripe values (line 63 in arrow.py). However, `_aggregate_files` assumes all stripe values are lists and calls `len()` on them (line 86), causing a TypeError when encountering `None`. This prevents users from combining `split_stripes=False` with `aggregate_files=True`.

## Fix

```diff
--- a/dask/dataframe/io/orc/arrow.py
+++ b/dask/dataframe/io/orc/arrow.py
@@ -82,7 +82,7 @@ class ArrowORCEngine:
     @classmethod
     def _aggregate_files(cls, aggregate_files, split_stripes, parts):
-        if aggregate_files is True and int(split_stripes) > 1 and len(parts) > 1:
+        if aggregate_files is True and split_stripes and int(split_stripes) > 1 and len(parts) > 1:
             new_parts = []
             new_part = parts[0]
             nstripes = len(new_part[0][1])
```