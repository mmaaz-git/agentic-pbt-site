# Bug Report: scipy.datasets.clear_cache Skips Validation When Cache Directory Doesn't Exist

**Target**: `scipy.datasets._utils.clear_cache`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `clear_cache()` function returns early when the cache directory doesn't exist, skipping validation of the `datasets` parameter. Invalid datasets that should raise `ValueError` are silently ignored.

## Property-Based Test

```python
import tempfile
import os
import pytest
from scipy.datasets._utils import _clear_cache
from scipy.datasets._registry import method_files_map


def test_clear_cache_validates_dataset_before_checking_cache_exists():
    def fake_dataset():
        pass
    fake_dataset.__name__ = "nonexistent_dataset"

    with tempfile.TemporaryDirectory() as tmpdir:
        nonexistent_cache = os.path.join(tmpdir, "nonexistent_cache_dir")

        with pytest.raises(ValueError, match="doesn't exist"):
            _clear_cache(
                datasets=[fake_dataset],
                cache_dir=nonexistent_cache,
                method_map=method_files_map
            )
```

**Failing input**: Invalid dataset callable when cache directory doesn't exist

## Reproducing the Bug

```python
import tempfile
import os
from scipy.datasets._utils import _clear_cache
from scipy.datasets._registry import method_files_map


def fake_dataset():
    pass
fake_dataset.__name__ = "nonexistent_dataset"

with tempfile.TemporaryDirectory() as tmpdir:
    nonexistent_cache = os.path.join(tmpdir, "nonexistent")

    _clear_cache(
        datasets=[fake_dataset],
        cache_dir=nonexistent_cache,
        method_map=method_files_map
    )
```

Expected: `ValueError` raised about invalid dataset
Actual: Function returns silently without validation

## Why This Is A Bug

The function's contract (lines 38-42 in `_utils.py`) states it should raise `ValueError` for invalid datasets. The early return at line 26 when the cache doesn't exist bypasses this validation, creating inconsistent behavior: invalid datasets raise errors when the cache exists but are silently ignored when it doesn't.

## Fix

```diff
--- a/scipy/datasets/_utils.py
+++ b/scipy/datasets/_utils.py
@@ -21,16 +21,8 @@ def _clear_cache(datasets, cache_dir=None, method_map=None):
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
             datasets = [datasets, ]
         for dataset in datasets:
             assert callable(dataset)
             dataset_name = dataset.__name__
             if dataset_name not in method_map:
                 raise ValueError(f"Dataset method {dataset_name} doesn't "
                                  "exist. Please check if the passed dataset "
                                  "is a subset of the following dataset "
                                  f"methods: {list(method_map.keys())}")

             data_files = method_map[dataset_name]
             data_filepaths = [os.path.join(cache_dir, file)
                               for file in data_files]
             for data_filepath in data_filepaths:
                 if os.path.exists(data_filepath):
                     print("Cleaning the file "
                           f"{os.path.split(data_filepath)[1]} "
                           f"for dataset {dataset_name}")
                     os.remove(data_filepath)
                 else:
                     print(f"Path {data_filepath} doesn't exist. "
                           "Nothing to clear.")
```