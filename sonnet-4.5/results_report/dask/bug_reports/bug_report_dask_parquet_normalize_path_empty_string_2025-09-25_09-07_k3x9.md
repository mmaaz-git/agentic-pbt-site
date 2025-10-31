# Bug Report: dask.dataframe.io.parquet _normalize_and_strip_protocol Empty String

**Target**: `dask.dataframe.dask_expr.io.parquet._normalize_and_strip_protocol`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_normalize_and_strip_protocol` function returns empty strings for valid filesystem root paths like "/" or "///", which are invalid paths that will cause downstream failures.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings, example
from dask.dataframe.dask_expr.io.parquet import _normalize_and_strip_protocol

@given(st.text(min_size=1))
@settings(max_examples=500)
@example("/")
@example("///")
@example("s3:///")
def test_normalize_and_strip_protocol_no_empty_strings(path):
    result = _normalize_and_strip_protocol(path)

    assert len(result) == 1
    assert result[0] != "", f"Result should not be empty string for input {path!r}"
```

**Failing input**: `"/"`

## Reproducing the Bug

```python
from dask.dataframe.dask_expr.io.parquet import _normalize_and_strip_protocol

result = _normalize_and_strip_protocol("/")
print(f"Input: '/'")
print(f"Output: {result!r}")
assert result == [''], "Returns empty string instead of valid path"

result = _normalize_and_strip_protocol("///")
print(f"Input: '///'")
print(f"Output: {result!r}")
assert result == [''], "Returns empty string for path with only slashes"

result = _normalize_and_strip_protocol(["s3://bucket/", "/"])
print(f"Input: ['s3://bucket/', '/']")
print(f"Output: {result!r}")
assert '' in result, "Contains empty string in result list"
```

## Why This Is A Bug

The function's purpose is to normalize paths by stripping protocol prefixes and trailing slashes. However, when a path consists entirely of slashes (e.g., "/", "///", "s3:///"), the function uses `rstrip("/")` which removes all slashes, resulting in an empty string.

This violates the following expected behaviors:
1. "/" is a valid filesystem path (root directory) and should remain valid after normalization
2. Empty strings are not valid paths and will cause failures in downstream PyArrow operations
3. The function should normalize paths, not eliminate them

The function is used in `ReadParquetPyarrowFS.normalized_path` (parquet.py:866), and the normalized path is later passed to `pa_fs.FileSelector(path, recursive=True)` (parquet.py:1019), which will fail or behave unexpectedly with empty strings.

## Fix

```diff
--- a/dask/dataframe/dask_expr/io/parquet.py
+++ b/dask/dataframe/dask_expr/io/parquet.py
@@ -1829,7 +1829,11 @@ def _normalize_and_strip_protocol(path):
             split = p.split(sep, 1)
             if len(split) > 1:
                 p = split[1]
                 break
-        result.append(p.rstrip("/"))
+        # Strip trailing slashes but preserve root path
+        normalized = p.rstrip("/")
+        if not normalized and p:
+            # If stripping resulted in empty string but input wasn't empty, keep as root
+            normalized = "/"
+        result.append(normalized)
     return result
```