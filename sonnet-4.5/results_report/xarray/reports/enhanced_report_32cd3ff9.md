# Bug Report: xarray.core.formatting_html._load_static_files Cache Corruption via Mutable Return Value

**Target**: `xarray.core.formatting_html._load_static_files`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_load_static_files()` function returns a cached mutable list object, allowing any caller to corrupt the cache by modifying the returned list, affecting all subsequent function calls.

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


if __name__ == "__main__":
    test_cache_returns_independent_copies()
```

<details>

<summary>
**Failing input**: `mutation_value=''` (or any other string value)
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/18/hypo.py", line 16, in <module>
    test_cache_returns_independent_copies()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/18/hypo.py", line 6, in test_cache_returns_independent_copies
    def test_cache_returns_independent_copies(mutation_value):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/18/hypo.py", line 12, in test_cache_returns_independent_copies
    assert len(call2) == 2
           ^^^^^^^^^^^^^^^
AssertionError
Falsifying example: test_cache_returns_independent_copies(
    mutation_value='',  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
from xarray.core.formatting_html import _load_static_files

# First call to _load_static_files()
first_call = _load_static_files()
print(f"First call: {len(first_call)} items")
print(f"Type: {type(first_call)}")

# Mutate the cached list by appending an item
first_call.append("INJECTED")
print(f"After mutation: {len(first_call)} items")

# Second call to _load_static_files()
second_call = _load_static_files()
print(f"Second call: {len(second_call)} items")

# Check if they're the same object
print(f"Same object: {first_call is second_call}")

# Show the last item to prove mutation persisted
print(f"Last item: {repr(second_call[-1])}")

# Third call to verify permanence
third_call = _load_static_files()
print(f"Third call: {len(third_call)} items")
```

<details>

<summary>
Cache corruption demonstrated - mutable list allows modification of cached data
</summary>
```
First call: 2 items
Type: <class 'list'>
After mutation: 3 items
Second call: 3 items
Same object: True
Last item: 'INJECTED'
Third call: 3 items
```
</details>

## Why This Is A Bug

This violates fundamental caching principles and Python's `lru_cache` best practices. According to Python's functools documentation, `lru_cache` caches the actual object reference, not a copy. The documentation warns that caching "doesn't make sense" for functions that "need to create distinct mutable objects on each call."

The function's name includes "static", implying the data should be immutable. The function loads CSS and HTML content from static resource files that should remain constant throughout the program's execution. The `@lru_cache(None)` decorator is intended to avoid redundant file I/O operations, not to share a mutable object across all callers.

When any code modifies the returned list (by appending, removing, or modifying elements), it corrupts the cache permanently. This affects all subsequent calls to `_load_static_files()`, potentially breaking the HTML/CSS rendering for all xarray objects displayed in notebooks or other HTML contexts.

## Relevant Context

The function is located at `/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages/xarray/core/formatting_html.py:29-35` and loads two static resource files defined in `STATIC_FILES`:
- `xarray.static.html/icons-svg-inline.html`
- `xarray.static.css/style.css`

The function is currently only called once in the codebase at line 305 in the `_obj_repr()` function, where the results are unpacked: `icons_svg, css_style = _load_static_files()`. This usage pattern doesn't modify the list, so the bug is latent in current code but represents a correctness issue that could manifest if the code evolves.

Python's official documentation for `functools.lru_cache`: https://docs.python.org/3/library/functools.html#functools.lru_cache

## Proposed Fix

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