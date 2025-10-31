# Bug Report: xarray.core.formatting_html._load_static_files Cache Mutation Vulnerability

**Target**: `xarray.core.formatting_html._load_static_files`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_load_static_files()` function uses `@lru_cache(None)` decorator but returns a mutable list, allowing callers to corrupt the shared cache by modifying the returned list, affecting all subsequent calls.

## Property-Based Test

```python
#!/usr/bin/env python3
"""
Hypothesis-based property test for xarray._load_static_files cache mutation bug.
This tests the invariant that cached values should not be mutable.
"""

from hypothesis import given, strategies as st
from xarray.core.formatting_html import _load_static_files

@given(st.text(min_size=1, max_size=100))
def test_cache_cannot_be_corrupted(corruption_text):
    """Test that the cache cannot be corrupted by modifying returned values."""
    # Clear cache to ensure clean state
    _load_static_files.cache_clear()

    # Get the first result and store the original value
    first = _load_static_files()
    original = first[0]

    # Try to corrupt the cache by modifying the returned list
    first[0] = corruption_text

    # Get the result again - it should be unchanged (immutable)
    second = _load_static_files()

    # The cache should NOT have been corrupted
    assert second[0] == original, f"Cache was corrupted! Expected original value but got: {corruption_text}"

if __name__ == "__main__":
    # Run the test with Hypothesis
    import sys
    try:
        test_cache_cannot_be_corrupted()
        print("Test passed! (This shouldn't happen if the bug exists)")
    except AssertionError as e:
        print(f"Test failed as expected: {e}")
        sys.exit(1)
```

<details>

<summary>
**Failing input**: `'0'`
</summary>
```
Test failed as expected: Cache was corrupted! Expected original value but got: 0
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""
Minimal reproduction of the xarray._load_static_files cache mutation bug.
This demonstrates that the cached list can be corrupted by callers.
"""

from xarray.core.formatting_html import _load_static_files

# Get the original cached values
print("Getting the original cached values...")
original = _load_static_files()
print(f"Type of returned value: {type(original)}")
print(f"Number of elements: {len(original)}")
print(f"First element starts with: {original[0][:50]}...")
print(f"Second element starts with: {original[1][:50]}...")

# Store the original first element for comparison
original_first_element = original[0]

# Corrupt the cache by modifying the returned list
print("\nCorrupting the cache by modifying the list...")
original[0] = "CORRUPTED_CACHE_VALUE"

# Get the value again - it should be immutable but it's not
print("\nGetting the cached values again...")
second = _load_static_files()
print(f"First element is now: {second[0]}")
print(f"Are they the same object? {original is second}")

# Verify the corruption
print("\nVerifying the corruption...")
if second[0] == "CORRUPTED_CACHE_VALUE":
    print("BUG CONFIRMED: The cached value was corrupted!")
    print("The function returns a mutable list that shares state across calls.")
else:
    print("No bug: The cached value was not corrupted.")
```

<details>

<summary>
Output showing cache corruption
</summary>
```
Getting the original cached values...
Type of returned value: <class 'list'>
Number of elements: 2
First element starts with: <svg style="position: absolute; width: 0; height: ...
Second element starts with: /* CSS stylesheet for displaying xarray objects in...

Corrupting the cache by modifying the list...

Getting the cached values again...
First element is now: CORRUPTED_CACHE_VALUE
Are they the same object? True

Verifying the corruption...
BUG CONFIRMED: The cached value was corrupted!
The function returns a mutable list that shares state across calls.
```
</details>

## Why This Is A Bug

This violates a fundamental principle of caching: cached values should be immutable. The Python documentation for `functools.lru_cache` warns against caching functions that create "distinct mutable objects on each call" - this is exactly what's happening here. The function returns the same mutable list object from cache, allowing any caller to modify it.

Specifically:
1. The function is decorated with `@lru_cache(None)` which caches the return value indefinitely
2. The function returns a list (mutable) containing static HTML and CSS resources
3. The same list object is returned on every call due to caching
4. Any code that calls `_load_static_files()` and modifies the returned list will corrupt the cache
5. All subsequent calls will receive the corrupted values until the cache is cleared
6. This could break HTML representations of xarray objects in Jupyter notebooks

The function name `_load_static_files` and its docstring "Lazily load the resource files into memory the first time they are needed" clearly indicate that these are meant to be static, unchanging resources that should not be modified after loading.

## Relevant Context

The `_load_static_files()` function is located in `/xarray/core/formatting_html.py` at lines 29-35. It's an internal function (underscore-prefixed) used to load static HTML and CSS resources for displaying xarray objects in Jupyter notebooks.

The function is called in the `_obj_repr()` function at line 305:
```python
icons_svg, css_style = _load_static_files()
```

These resources are then embedded into the HTML representation of xarray objects. If the cache is corrupted, all xarray HTML visualizations in the current Python session would be broken.

Python's documentation on `lru_cache`: https://docs.python.org/3/library/functools.html#functools.lru_cache

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