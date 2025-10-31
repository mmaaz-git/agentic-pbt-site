# Bug Report: scipy.datasets.clear_cache Input Validation

**Target**: `scipy.datasets.clear_cache`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`scipy.datasets.clear_cache()` silently accepts invalid input types (strings, integers, dicts, etc.) when the cache directory doesn't exist, violating its documented API contract that specifies it should only accept `None`, a callable, or a list/tuple of callables.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pytest
import scipy.datasets

@given(st.text(min_size=1))
def test_clear_cache_rejects_arbitrary_strings(text_input):
    """Property: clear_cache should reject any string input"""
    if callable(text_input):
        return

    with pytest.raises((ValueError, TypeError, AssertionError)):
        scipy.datasets.clear_cache(text_input)

@given(st.integers())
def test_clear_cache_rejects_integers(int_input):
    """Property: clear_cache should reject integer inputs"""
    with pytest.raises((ValueError, TypeError, AssertionError)):
        scipy.datasets.clear_cache(int_input)
```

**Failing input**: `'0'` (or any string, integer, dict, etc.)

## Reproducing the Bug

```python
import scipy.datasets

scipy.datasets.clear_cache("invalid_string")
scipy.datasets.clear_cache(42)
scipy.datasets.clear_cache({"key": "value"})
```

Output:
```
Cache Directory /home/npc/.cache/scipy-data doesn't exist. Nothing to clear.
```

All three invalid inputs are silently accepted without raising an error, violating the documented contract.

## Why This Is A Bug

According to the function's docstring:
```
Parameters
----------
datasets : callable or list/tuple of callable or None
```

The function should only accept:
1. `None` (to clear all caches)
2. A callable (e.g., `scipy.datasets.ascent`)
3. A list/tuple of callables

However, the implementation in `/home/npc/.local/lib/python3.13/site-packages/scipy/datasets/_utils.py` has a logic error where it returns early when the cache directory doesn't exist (line 19-21), **before** performing input validation (line 31).

This creates inconsistent behavior:
- When cache directory **doesn't exist**: Invalid inputs silently accepted
- When cache directory **exists**: Invalid inputs properly rejected with `AssertionError`

This violates the principle of fail-fast and could mask user errors.

## Fix

```diff
--- a/scipy/datasets/_utils.py
+++ b/scipy/datasets/_utils.py
@@ -16,12 +16,24 @@ def _clear_cache(datasets, cache_dir=None, method_map=None):
                               "conda to install 'pooch'.")
         cache_dir = platformdirs.user_cache_dir("scipy-data")

+    # Validate input before checking cache directory existence
+    if datasets is not None:
+        if not isinstance(datasets, list | tuple):
+            datasets = [datasets, ]
+        for dataset in datasets:
+            if not callable(dataset):
+                raise TypeError(
+                    f"datasets parameter must be None, a callable, or a "
+                    f"list/tuple of callables, got {type(dataset).__name__}"
+                )
+
     if not os.path.exists(cache_dir):
         print(f"Cache Directory {cache_dir} doesn't exist. Nothing to clear.")
         return

     if datasets is None:
         print(f"Cleaning the cache directory {cache_dir}!")
         shutil.rmtree(cache_dir)
     else:
-        if not isinstance(datasets, list | tuple):
-            datasets = [datasets, ]
         for dataset in datasets:
-            assert callable(dataset)
             dataset_name = dataset.__name__
```

This fix moves input validation before the cache directory existence check, ensuring consistent behavior and proper error messages regardless of system state.