# Bug Report: xarray.core.formatting_html._load_static_files() Cache Mutation Vulnerability

**Target**: `xarray.core.formatting_html._load_static_files()`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_load_static_files()` function uses `@lru_cache(None)` but returns a mutable list, allowing callers to modify the shared cached object and affect all future calls.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from xarray.core.formatting_html import _load_static_files

@given(st.text())
def test_load_static_files_immutability(injection):
    result1 = _load_static_files()
    original_len = len(result1)

    result1.append(injection)

    result2 = _load_static_files()

    assert len(result2) == original_len, "Cached list should not be mutated"
```

**Failing input**: Any string, e.g., `"INJECTED"`

## Reproducing the Bug

```python
from xarray.core.formatting_html import _load_static_files

result1 = _load_static_files()
original_len = len(result1)

result1.append("MALICIOUS_INJECTION")

result2 = _load_static_files()
assert len(result2) == original_len + 1
assert result2 is result1
```

## Why This Is A Bug

The `@lru_cache` decorator caches the return value of `_load_static_files()`, which is a list. Since lists are mutable in Python, any caller can modify this shared list, affecting all subsequent calls. This violates the principle that cached functions should return immutable data or copies to prevent cache poisoning.

While unlikely to be exploited in practice (since callers don't typically modify the returned list), this is still a logic error that could lead to subtle bugs if the list is accidentally modified.

## Fix

```diff
--- a/xarray/core/formatting_html.py
+++ b/xarray/core/formatting_html.py
@@ -29,7 +29,7 @@ if TYPE_CHECKING:
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

Alternatively, return a copy on each call:

```diff
--- a/xarray/core/formatting_html.py
+++ b/xarray/core/formatting_html.py
@@ -27,10 +27,14 @@ if TYPE_CHECKING:


 @lru_cache(None)
-def _load_static_files():
+def _load_static_files_cached():
     """Lazily load the resource files into memory the first time they are needed"""
     return [
         files(package).joinpath(resource).read_text(encoding="utf-8")
         for package, resource in STATIC_FILES
     ]
+
+
+def _load_static_files():
+    return list(_load_static_files_cached())
```