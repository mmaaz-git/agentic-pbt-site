# Bug Report: xarray.core.formatting_html._load_static_files Cache Mutation

**Target**: `xarray.core.formatting_html._load_static_files`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_load_static_files()` function uses `@lru_cache` and returns a mutable list, allowing callers to mutate the cached value and affect all subsequent callers.

## Property-Based Test

```python
from hypothesis import given
from xarray.core import formatting_html as fh


def test_load_static_files_cache_immutability():
    first_call = fh._load_static_files()
    original_value = first_call[0]

    second_call = fh._load_static_files()
    second_call[0] = "MUTATED"

    third_call = fh._load_static_files()

    assert third_call[0] == original_value, "Cache should not be mutated by callers"
```

**Failing input**: Any call sequence that modifies the returned list

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/path/to/xarray')

from xarray.core import formatting_html as fh

first = fh._load_static_files()
print(f"Original first element (truncated): {first[0][:50]}")

first[0] = "MUTATED"

second = fh._load_static_files()
print(f"After mutation: {second[0]}")

assert second[0] == "MUTATED", "BUG: Cache was mutated!"
```

## Why This Is A Bug

The `_load_static_files()` function is decorated with `@lru_cache(None)` on line 29 and returns a list on lines 32-35. Since `lru_cache` returns the same object on subsequent calls and lists are mutable in Python, any caller can modify the cached list, affecting all subsequent callers. This violates the expectation that cached values are stable and immutable.

This is a well-known anti-pattern in Python when using `lru_cache` with mutable return types.

## Fix

```diff
--- a/xarray/core/formatting_html.py
+++ b/xarray/core/formatting_html.py
@@ -29,7 +29,7 @@ STATIC_FILES = (
 @lru_cache(None)
 def _load_static_files():
     """Lazily load the resource files into memory the first time they are needed"""
-    return [
+    return tuple(
         files(package).joinpath(resource).read_text(encoding="utf-8")
         for package, resource in STATIC_FILES
-    ]
+    )
```

By returning a `tuple` instead of a `list`, the return value becomes immutable, preventing cache pollution while maintaining the same functionality (both support indexing and unpacking).