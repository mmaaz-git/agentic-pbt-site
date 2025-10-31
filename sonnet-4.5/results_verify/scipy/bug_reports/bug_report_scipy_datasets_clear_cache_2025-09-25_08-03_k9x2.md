# Bug Report: scipy.datasets.clear_cache Input Validation Bypass

**Target**: `scipy.datasets._utils._clear_cache`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `_clear_cache` function fails to validate dataset method inputs when the cache directory doesn't exist, violating its documented API contract.

## Property-Based Test

```python
import os
import tempfile
import pytest
from hypothesis import given, strategies as st
from scipy.datasets._utils import _clear_cache

def make_invalid_dataset(name):
    def invalid():
        pass
    invalid.__name__ = name
    return invalid

@given(st.text(min_size=1).filter(lambda x: x not in ['ascent', 'electrocardiogram', 'face']))
def test_clear_cache_validates_dataset_regardless_of_cache_existence(invalid_name):
    invalid_dataset = make_invalid_dataset(invalid_name)

    with tempfile.TemporaryDirectory() as tmpdir:
        non_existent = os.path.join(tmpdir, "nonexistent")

        with pytest.raises(ValueError, match="doesn't exist"):
            _clear_cache(invalid_dataset, cache_dir=non_existent)
```

**Failing input**: `invalid_name='0'` (or any string not in `['ascent', 'electrocardiogram', 'face']`)

## Reproducing the Bug

```python
import os
import tempfile
from scipy.datasets._utils import _clear_cache

def invalid_dataset():
    pass

with tempfile.TemporaryDirectory() as tmpdir:
    non_existent = os.path.join(tmpdir, "nonexistent")

    _clear_cache(invalid_dataset, cache_dir=non_existent)
    print("No error raised! Expected ValueError.")

    _clear_cache(invalid_dataset, cache_dir=tmpdir)
```

Expected: ValueError raised in both cases
Actual: ValueError only raised when cache directory exists

## Why This Is A Bug

The function's early return on line 24-26 of `_utils.py` bypasses input validation (lines 36-42). This creates inconsistent behavior:

1. With existing cache: ValueError raised for invalid dataset methods
2. With non-existent cache: No error, silently returns

This violates the principle of fail-fast and makes the API behavior dependent on filesystem state rather than input correctness. Users cannot rely on the function to catch invalid inputs.

## Fix

```diff
--- a/scipy/datasets/_utils.py
+++ b/scipy/datasets/_utils.py
@@ -21,15 +21,19 @@ def _clear_cache(datasets, cache_dir=None, method_map=None):
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
         if not isinstance(datasets, list | tuple):
             # single dataset method passed should be converted to list
             datasets = [datasets, ]
         for dataset in datasets:
             assert callable(dataset)
             dataset_name = dataset.__name__  # Name of the dataset method
             if dataset_name not in method_map:
                 raise ValueError(f"Dataset method {dataset_name} doesn't "
                                  "exist. Please check if the passed dataset "
                                  "is a subset of the following dataset "
                                  f"methods: {list(method_map.keys())}")

+        if not os.path.exists(cache_dir):
+            print(f"Cache Directory {cache_dir} doesn't exist. Nothing to clear.")
+            return
+
             data_files = method_map[dataset_name]
             data_filepaths = [os.path.join(cache_dir, file)
                               for file in data_files]
```