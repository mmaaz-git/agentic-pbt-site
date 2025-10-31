# Bug Report: scipy.datasets.clear_cache Input Validation Bypass

**Target**: `scipy.datasets.clear_cache`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`clear_cache()` skips input validation when the cache directory doesn't exist, silently accepting invalid inputs instead of raising appropriate errors.

## Property-Based Test

```python
import pytest
from hypothesis import given, strategies as st
import scipy.datasets


@given(st.text())
def test_clear_cache_rejects_non_callables(invalid_input):
    """clear_cache should reject non-callable inputs regardless of cache state"""
    if not callable(invalid_input):
        with pytest.raises((AssertionError, TypeError)):
            scipy.datasets.clear_cache(invalid_input)


def test_clear_cache_rejects_invalid_callables():
    """clear_cache should validate callable names regardless of cache state"""
    def invalid_dataset():
        pass

    with pytest.raises(ValueError, match="Dataset method .* doesn't exist"):
        scipy.datasets.clear_cache(invalid_dataset)
```

**Failing input**: Any invalid input when cache directory doesn't exist

## Reproducing the Bug

```python
import scipy.datasets

def invalid_dataset():
    pass

scipy.datasets.clear_cache("not_callable")

scipy.datasets.clear_cache(invalid_dataset)
```

Expected: Should raise `AssertionError` or `TypeError` for non-callable, and `ValueError` for invalid callable.

Actual: Prints "Cache Directory ... doesn't exist. Nothing to clear." and returns successfully.

## Why This Is A Bug

The function's contract (lines 32-42 in `_utils.py`) specifies that it should validate inputs - it has an assertion that `callable(dataset)` and raises `ValueError` for invalid dataset names. However, this validation is bypassed when the cache directory doesn't exist because the function returns early (line 26).

This violates the Fail Fast principle and API contract - invalid inputs should be rejected immediately, regardless of system state.

## Fix

Move input validation before the cache directory check:

```diff
--- a/scipy/datasets/_utils.py
+++ b/scipy/datasets/_utils.py
@@ -21,17 +21,24 @@ def _clear_cache(datasets, cache_dir=None, method_map=None):
                           "conda to install 'pooch'.")
         cache_dir = platformdirs.user_cache_dir("scipy-data")

-    if not os.path.exists(cache_dir):
-        print(f"Cache Directory {cache_dir} doesn't exist. Nothing to clear.")
-        return
-
     if datasets is None:
-        print(f"Cleaning the cache directory {cache_dir}!")
-        shutil.rmtree(cache_dir)
+        # Validate cache exists for None case
+        if not os.path.exists(cache_dir):
+            print(f"Cache Directory {cache_dir} doesn't exist. Nothing to clear.")
+            return
+        print(f"Cleaning the cache directory {cache_dir}!")
+        shutil.rmtree(cache_dir)
     else:
         if not isinstance(datasets, list | tuple):
-            # single dataset method passed should be converted to list
             datasets = [datasets, ]
+
+        # Validate inputs first
         for dataset in datasets:
             assert callable(dataset)
             dataset_name = dataset.__name__
@@ -41,6 +48,11 @@ def _clear_cache(datasets, cache_dir=None, method_map=None):
                                  "is a subset of the following dataset "
                                  f"methods: {list(method_map.keys())}")

+        # Now check if cache exists
+        if not os.path.exists(cache_dir):
+            print(f"Cache Directory {cache_dir} doesn't exist. Nothing to clear.")
+            return
+
             data_files = method_map[dataset_name]
             data_filepaths = [os.path.join(cache_dir, file)
                               for file in data_files]
```