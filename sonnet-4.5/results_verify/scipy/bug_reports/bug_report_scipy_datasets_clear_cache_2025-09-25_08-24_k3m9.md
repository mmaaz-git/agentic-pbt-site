# Bug Report: scipy.datasets.clear_cache Validation Bypass

**Target**: `scipy.datasets.clear_cache`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `clear_cache()` function fails to validate dataset arguments when the cache directory doesn't exist, allowing invalid dataset methods to pass without raising the documented ValueError.

## Property-Based Test

```python
import pytest
from scipy import datasets


def test_clear_cache_validates_before_checking_directory():
    def fake_dataset():
        pass

    with pytest.raises(ValueError, match="Dataset method fake_dataset doesn't exist"):
        datasets.clear_cache(fake_dataset)
```

**Failing input**: `fake_dataset` (any invalid dataset method) when cache directory doesn't exist

## Reproducing the Bug

```python
from scipy import datasets


def fake_dataset():
    pass


datasets.clear_cache(fake_dataset)
```

Expected: `ValueError: Dataset method fake_dataset doesn't exist...`
Actual: No error, prints "Cache Directory ... doesn't exist. Nothing to clear."

## Why This Is A Bug

The function's contract promises to validate dataset methods (as shown by the ValueError in the source code for invalid datasets). However, this validation is bypassed when the cache directory doesn't exist. This violates the fail-fast principle and makes the API inconsistent - the same invalid input either raises an error or silently succeeds depending on whether a cache directory exists.

## Fix

```diff
--- a/scipy/datasets/_utils.py
+++ b/scipy/datasets/_utils.py
@@ -20,17 +20,6 @@ def _clear_cache(datasets, cache_dir=None, method_map=None):
                               "conda to install 'pooch'.")
         cache_dir = platformdirs.user_cache_dir("scipy-data")

-    if not os.path.exists(cache_dir):
-        print(f"Cache Directory {cache_dir} doesn't exist. Nothing to clear.")
-        return
-
     if datasets is None:
-        print(f"Cleaning the cache directory {cache_dir}!")
-        shutil.rmtree(cache_dir)
+        if os.path.exists(cache_dir):
+            print(f"Cleaning the cache directory {cache_dir}!")
+            shutil.rmtree(cache_dir)
+        else:
+            print(f"Cache Directory {cache_dir} doesn't exist. Nothing to clear.")
     else:
         if not isinstance(datasets, list | tuple):
@@ -43,11 +32,14 @@ def _clear_cache(datasets, cache_dir=None, method_map=None):

             data_files = method_map[dataset_name]
             data_filepaths = [os.path.join(cache_dir, file)
                               for file in data_files]
+
+            if not os.path.exists(cache_dir):
+                print(f"Cache Directory {cache_dir} doesn't exist. Nothing to clear.")
+                continue
+
             for data_filepath in data_filepaths:
                 if os.path.exists(data_filepath):
                     print("Cleaning the file "
```