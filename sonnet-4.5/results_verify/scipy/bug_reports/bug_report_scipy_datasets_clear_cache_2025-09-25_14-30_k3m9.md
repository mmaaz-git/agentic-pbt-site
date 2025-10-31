# Bug Report: scipy.datasets.clear_cache Input Validation Bypass

**Target**: `scipy.datasets._utils._clear_cache`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `clear_cache` function fails to validate input arguments when the cache directory doesn't exist, accepting invalid inputs (strings, integers, dicts) that should raise an AssertionError or TypeError.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st
import pytest
import scipy.datasets

@settings(max_examples=10)
@given(st.sampled_from(["invalid", 123, 45.6, {"key": "value"}]))
def test_clear_cache_rejects_invalid_inputs(invalid_input):
    with pytest.raises((ValueError, TypeError, AssertionError)):
        scipy.datasets.clear_cache(invalid_input)
```

**Failing input**: `"invalid"` (or any non-callable value when cache directory doesn't exist)

## Reproducing the Bug

```python
import tempfile
import os
import shutil
from scipy.datasets._utils import _clear_cache

temp_cache = tempfile.mkdtemp()
non_existent_cache = os.path.join(temp_cache, "scipy-data-test")

print(f"Cache exists: {os.path.exists(non_existent_cache)}")

_clear_cache("invalid_string", cache_dir=non_existent_cache)
print("Bug: No exception raised for invalid input!")

shutil.rmtree(temp_cache)
```

## Why This Is A Bug

According to the function's code (line 36 in `_utils.py`), there's an `assert callable(dataset)` that validates inputs. However, the function returns early on line 26 when the cache directory doesn't exist, bypassing this validation. This violates the API contract - the function should reject invalid inputs regardless of cache state.

The documented API states that `datasets` parameter should be "callable or list/tuple of callable or None", but the function silently accepts any value when the cache doesn't exist.

## Fix

```diff
diff --git a/scipy/datasets/_utils.py b/scipy/datasets/_utils.py
index 1234567..abcdef0 100644
--- a/scipy/datasets/_utils.py
+++ b/scipy/datasets/_utils.py
@@ -21,14 +21,6 @@ def _clear_cache(datasets, cache_dir=None, method_map=None):
                               "conda to install 'pooch'.")
         cache_dir = platformdirs.user_cache_dir("scipy-data")

-    if not os.path.exists(cache_dir):
-        print(f"Cache Directory {cache_dir} doesn't exist. Nothing to clear.")
-        return
-
     if datasets is None:
+        if not os.path.exists(cache_dir):
+            print(f"Cache Directory {cache_dir} doesn't exist. Nothing to clear.")
+            return
         print(f"Cleaning the cache directory {cache_dir}!")
         shutil.rmtree(cache_dir)
     else:
@@ -36,6 +28,9 @@ def _clear_cache(datasets, cache_dir=None, method_map=None):
         for dataset in datasets:
             assert callable(dataset)
             dataset_name = dataset.__name__
+            if not os.path.exists(cache_dir):
+                print(f"Cache Directory {cache_dir} doesn't exist. Nothing to clear.")
+                return
             if dataset_name not in method_map:
                 raise ValueError(f"Dataset method {dataset_name} doesn't "
                                  "exist. Please check if the passed dataset "
```