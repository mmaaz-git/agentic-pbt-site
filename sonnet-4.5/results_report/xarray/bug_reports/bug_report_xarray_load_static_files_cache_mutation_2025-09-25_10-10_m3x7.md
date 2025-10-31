# Bug Report: xarray._load_static_files Cache Mutation

**Target**: `xarray.core.formatting_html._load_static_files`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_load_static_files()` function uses `@lru_cache` but returns a mutable list, allowing callers to corrupt the cached data. This violates the idempotence property expected from cached functions.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
from xarray.core.formatting_html import _load_static_files


@given(st.text())
def test_cache_immutability(mutation_data):
    result1 = _load_static_files()
    original_length = len(result1)

    result1.append(mutation_data)

    result2 = _load_static_files()

    assert len(result2) == original_length, \
        "Cached result should not be affected by mutations to previous return values"
```

**Failing input**: `any string` (e.g., `"test"`)

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

from xarray.core.formatting_html import _load_static_files

result1 = _load_static_files()
print(f"Initial length: {len(result1)}")

result1.append("CORRUPTED DATA")

result2 = _load_static_files()
print(f"Length after mutation: {len(result2)}")

assert len(result2) == len(result1), "Cache was mutated!"
```

## Why This Is A Bug

The `@lru_cache` decorator is designed to improve performance by caching function results. However, when a cached function returns a mutable object (like a list), callers can inadvertently or maliciously modify the cached data, affecting all future calls. This violates the expected behavior that cached functions return consistent, unmodified results.

While this may seem like a low-probability issue, it could lead to subtle bugs if any code path accidentally modifies the returned list (e.g., with `.append()`, `.pop()`, or item assignment).

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

The fix changes the return type from a mutable `list` to an immutable `tuple`. Since the function returns static file contents that never change, using a tuple is more appropriate and prevents cache corruption.