# Bug Report: xarray.static _load_static_files Cache Mutability

**Target**: `xarray.core.formatting_html._load_static_files`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_load_static_files()` function returns a cached mutable list, allowing callers to inadvertently corrupt the cache by modifying the returned list.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from xarray.core.formatting_html import _load_static_files


@given(st.text())
def test_cache_returns_independent_copies(mutation_value):
    call1 = _load_static_files()
    call2 = _load_static_files()

    call1.append(mutation_value)

    assert len(call2) == 2
```

**Failing input**: Any string value (e.g., `"test"`)

## Reproducing the Bug

```python
from xarray.core.formatting_html import _load_static_files

first_call = _load_static_files()
print(f"First call: {len(first_call)} items")

first_call.append("INJECTED")

second_call = _load_static_files()
print(f"Second call: {len(second_call)} items")
print(f"Same object: {first_call is second_call}")
print(f"Last item: {second_call[-1]}")
```

## Why This Is A Bug

The `@lru_cache(None)` decorator caches the returned list object. Since Python lists are mutable, any caller that modifies the returned list will corrupt the cache for all subsequent calls. This violates the expected behavior of cached functions, which should return consistent results.

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