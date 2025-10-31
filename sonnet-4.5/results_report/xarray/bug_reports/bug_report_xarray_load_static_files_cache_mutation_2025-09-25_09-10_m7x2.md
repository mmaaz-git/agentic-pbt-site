# Bug Report: xarray.core.formatting_html._load_static_files Cache Mutation

**Target**: `xarray.core.formatting_html._load_static_files`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_load_static_files()` function uses `@lru_cache(None)` but returns a mutable list. Callers can corrupt the cache by modifying the returned list, affecting all subsequent calls.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from xarray.core.formatting_html import _load_static_files

@given(st.text(min_size=1, max_size=100))
def test_cache_cannot_be_corrupted(corruption_text):
    _load_static_files.cache_clear()
    first = _load_static_files()
    original = first[0]

    first[0] = corruption_text
    second = _load_static_files()

    assert second[0] == original
```

**Failing input**: `'0'` (or any text)

## Reproducing the Bug

```python
from xarray.core.formatting_html import _load_static_files

original = _load_static_files()
print(f"Original: {original[0][:50]}")

original[0] = "CORRUPTED"

second = _load_static_files()
print(f"After mutation: {second[0]}")

assert second[0] == "CORRUPTED"
```

## Why This Is A Bug

When a cached function returns a mutable object, callers can accidentally or maliciously modify the cached value, affecting all future callers. This violates the fundamental contract of caching: the function should return consistent, immutable results.

In this case, if any code modifies the list returned by `_load_static_files()`, all HTML representations in xarray will be corrupted, potentially breaking visualizations in Jupyter notebooks.

## Fix

```diff
--- a/xarray/core/formatting_html.py
+++ b/xarray/core/formatting_html.py
@@ -29,7 +29,7 @@ STATIC_FILES = (
 @lru_cache(None)
 def _load_static_files():
     """Lazily load the resource files into memory the first time they are needed"""
     return [
         files(package).joinpath(resource).read_text(encoding="utf-8")
         for package, resource in STATIC_FILES
     ]
+    return tuple(
+        files(package).joinpath(resource).read_text(encoding="utf-8")
+        for package, resource in STATIC_FILES
+    )
```