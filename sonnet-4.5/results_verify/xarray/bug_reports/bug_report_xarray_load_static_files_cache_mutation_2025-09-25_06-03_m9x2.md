# Bug Report: xarray.core.formatting_html._load_static_files Cache Mutation

**Target**: `xarray.core.formatting_html._load_static_files`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_load_static_files()` function uses `@lru_cache(None)` to cache static file contents, but returns a mutable list. If any caller modifies the returned list, all subsequent calls receive the mutated cached value, violating the function's invariant of always returning the original static file contents.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
from xarray.core.formatting_html import _load_static_files


@given(st.integers(min_value=0, max_value=1))
def test_load_static_files_cache_mutation(index):
    result1 = _load_static_files()
    original_first_item = result1[0]

    result1[index] = "MUTATED"

    result2 = _load_static_files()

    assert result2[0] == original_first_item, "Cache returned mutated result"
```

**Failing input**: `index=0`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

from xarray.core.formatting_html import _load_static_files

result1 = _load_static_files()
print(f"First call: {result1[0][:50]}...")

result1[0] = "MUTATED_CONTENT"

result2 = _load_static_files()
print(f"Second call: {result2[0]}")

assert result2[0] == "MUTATED_CONTENT"
```

## Why This Is A Bug

The function is supposed to lazily load and cache the static HTML/CSS files, ensuring all callers receive the same immutable content. However, because it returns a mutable list, any caller that modifies the returned value corrupts the cache for all future calls. This violates the expected behavior of a cached resource loader and could cause:

1. HTML rendering failures if the SVG/CSS content is corrupted
2. Unpredictable behavior depending on call order
3. Hard-to-debug issues since the mutation may happen in unrelated code

The docstring states "Lazily load the resource files into memory the first time they are needed", implying the content should remain stable and unchanged.

## Fix

Return a tuple instead of a list to prevent mutation:

```diff
--- a/xarray/core/formatting_html.py
+++ b/xarray/core/formatting_html.py
@@ -29,9 +29,9 @@ STATIC_FILES = (
 @lru_cache(None)
 def _load_static_files():
     """Lazily load the resource files into memory the first time they are needed"""
-    return [
+    return tuple([
         files(package).joinpath(resource).read_text(encoding="utf-8")
         for package, resource in STATIC_FILES
-    ]
+    ])
```