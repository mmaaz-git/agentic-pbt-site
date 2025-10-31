# Bug Report: dask.dataframe.io.parquet _normalize_and_strip_protocol Empty String

**Target**: `dask.dataframe.io.parquet._normalize_and_strip_protocol`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_normalize_and_strip_protocol` function can return a list containing empty strings when given paths that consist only of slashes (e.g., `"/"`, `"///"`, or protocol-only paths like `"file:///"`). This violates the reasonable expectation that normalized paths should be non-empty strings.

## Property-Based Test

```python
from hypothesis import given, strategies as st, example

@given(st.lists(st.text(min_size=0), min_size=1))
@example(["/"])
@example(["///"])
@example(["file:///"])
@example(["s3:///"])
def test_no_empty_strings_in_normalized_paths(paths):
    from dask.dataframe.dask_expr.io.parquet import _normalize_and_strip_protocol
    result = _normalize_and_strip_protocol(paths)
    for r in result:
        assert r != "", f"Empty string found in result for input {paths}"
```

**Failing inputs**: `["/"]`, `["///"]`, `["file:///"]`, `["s3:///"]`

## Reproducing the Bug

```python
from dask.dataframe.dask_expr.io.parquet import _normalize_and_strip_protocol

assert _normalize_and_strip_protocol("/") == [""]
assert _normalize_and_strip_protocol("///") == [""]
assert _normalize_and_strip_protocol("file:///") == [""]
assert _normalize_and_strip_protocol("s3:///") == [""]
```

## Why This Is A Bug

1. Empty strings are not valid file paths in most filesystem contexts
2. The function is used to normalize paths for parquet file operations, where empty paths would cause errors
3. This violates the invariant that path normalization should preserve the semantic meaning of paths
4. The docstring and usage context suggest paths should be valid filesystem references

## Fix

The issue occurs at line 1834 where `rstrip("/")` can produce empty strings. The function should preserve at least the root path `/` or handle empty results appropriately:

```diff
--- a/dask/dataframe/dask_expr/io/parquet.py
+++ b/dask/dataframe/dask_expr/io/parquet.py
@@ -1831,7 +1831,10 @@ def _normalize_and_strip_protocol(path):
             if len(split) > 1:
                 p = split[1]
                 break
-        result.append(p.rstrip("/"))
+        normalized = p.rstrip("/")
+        if not normalized:
+            normalized = "/"
+        result.append(normalized)
     return result
```

Alternatively, the function could raise an error for invalid inputs, or document that empty strings are possible return values.