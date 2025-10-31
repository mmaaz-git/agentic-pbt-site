# Bug Report: _normalize_and_strip_protocol Lacks Idempotence

**Target**: `dask.dataframe.dask_expr.io.parquet._normalize_and_strip_protocol`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_normalize_and_strip_protocol` function is not idempotent when paths contain multiple `::` protocol separators. Calling the function multiple times on the same input produces different results, which violates the expected behavior of a normalization function.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from dask.dataframe.dask_expr.io.parquet import _normalize_and_strip_protocol

@given(st.lists(st.text(min_size=1), min_size=1))
def test_normalize_path_idempotence(paths):
    result1 = _normalize_and_strip_protocol(paths)
    result2 = _normalize_and_strip_protocol(result1)
    assert result1 == result2, f"Idempotence violated: {result1} != {result2}"
```

**Failing input**: `["cache::s3::bucket/path"]`

## Reproducing the Bug

```python
from dask.dataframe.dask_expr.io.parquet import _normalize_and_strip_protocol

path = "cache::s3::bucket/path"
print(f"Input: {path}")

result1 = _normalize_and_strip_protocol(path)
print(f"After 1st call: {result1}")

result2 = _normalize_and_strip_protocol(result1)
print(f"After 2nd call: {result2}")

result3 = _normalize_and_strip_protocol(result2)
print(f"After 3rd call: {result3}")

assert result2 == result3, "Idempotence violated!"
```

Output:
```
Input: cache::s3::bucket/path
After 1st call: ['s3::bucket/path']
After 2nd call: ['bucket/path']
After 3rd call: ['bucket/path']
```

## Why This Is A Bug

A normalization function should be idempotent: applying it multiple times should produce the same result as applying it once. The function `_normalize_and_strip_protocol` violates this property because:

1. The first call strips the first `::` separator: `"cache::s3::bucket/path"` → `["s3::bucket/path"]`
2. The second call strips another `::` separator: `["s3::bucket/path"]` → `["bucket/path"]`
3. The third call produces the same result: `["bucket/path"]` → `["bucket/path"]`

This means `f(x) ≠ f(f(x))` for inputs containing multiple `::` separators.

This is problematic because:
- The function is used in `ReadParquetPyarrowFS.normalized_path` to normalize file paths
- Paths with chained fsspec protocols (e.g., `"simplecache::s3://..."`) are valid and commonly used
- Multiple normalizations could occur through caching or optimization passes
- Different code paths might normalize the path a different number of times, leading to inconsistent behavior

## Fix

The function should normalize the path completely in a single pass. The current implementation uses a `break` statement that stops after finding the first protocol separator. The fix should remove all protocol separators in one iteration:

```diff
diff --git a/dask/dataframe/dask_expr/io/parquet.py b/dask/dataframe/dask_expr/io/parquet.py
index abc123..def456 100644
--- a/dask/dataframe/dask_expr/io/parquet.py
+++ b/dask/dataframe/dask_expr/io/parquet.py
@@ -1822,13 +1822,11 @@ def _normalize_and_strip_protocol(path):
     if not isinstance(path, (list, tuple)):
         path = [path]

     result = []
     for p in path:
         protocol_separators = ["://", "::"]
         for sep in protocol_separators:
-            split = p.split(sep, 1)
-            if len(split) > 1:
-                p = split[1]
-                break
+            while sep in p:
+                p = p.split(sep, 1)[1]
         result.append(p.rstrip("/"))
     return result
```

This ensures all occurrences of both `://` and `::` are stripped, making the function truly idempotent.