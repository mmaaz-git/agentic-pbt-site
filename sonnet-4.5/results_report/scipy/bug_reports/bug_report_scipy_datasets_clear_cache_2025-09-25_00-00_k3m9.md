# Bug Report: scipy.datasets.clear_cache Input Validation Bypass

**Target**: `scipy.datasets.clear_cache` (specifically `scipy.datasets._utils._clear_cache`)
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `clear_cache` function accepts invalid inputs (non-callables) without raising an error when the cache directory doesn't exist, violating its documented contract that it should only accept `callable or list/tuple of callable or None`.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
import pytest
import scipy.datasets


@given(st.one_of(st.integers(), st.text(), st.floats()))
def test_clear_cache_non_callable_handling(value):
    assume(not callable(value))

    with pytest.raises((AssertionError, ValueError, TypeError)):
        scipy.datasets.clear_cache(value)
```

**Failing input**: Any non-callable value (e.g., `42`, `"string"`, `3.14`) when the cache directory doesn't exist

## Reproducing the Bug

```python
import scipy.datasets._utils as utils

cache_path = "/tmp/nonexistent_scipy_cache_xyz"

utils._clear_cache(datasets=42, cache_dir=cache_path)

utils._clear_cache(datasets="not a callable", cache_dir=cache_path)

utils._clear_cache(datasets=[1, 2, 3], cache_dir=cache_path)
```

Expected: `AssertionError` or `TypeError` should be raised for invalid inputs.
Actual: No exception is raised; the function silently returns after printing "Cache Directory ... doesn't exist. Nothing to clear."

## Why This Is A Bug

The function's docstring specifies that the `datasets` parameter should be `callable or list/tuple of callable or None`. The function contains an `assert callable(dataset)` check (line 36 of `_utils.py`), indicating that the developers intended to validate inputs.

However, this validation is bypassed when the cache directory doesn't exist because of an early return (lines 24-26). This creates inconsistent behavior: the same invalid input is accepted when the cache doesn't exist but rejected when it does exist.

This violates the API contract and the principle of fail-fast validation. Functions should validate their inputs regardless of the system state.

## Fix

Move the input validation logic before the cache existence check:

```diff
--- a/scipy/datasets/_utils.py
+++ b/scipy/datasets/_utils.py
@@ -21,6 +21,24 @@ def _clear_cache(datasets, cache_dir=None, method_map=None):
                               "conda to install 'pooch'.")
         cache_dir = platformdirs.user_cache_dir("scipy-data")

+    # Validate inputs before checking cache existence
+    if datasets is not None:
+        if not isinstance(datasets, list | tuple):
+            datasets = [datasets, ]
+        for dataset in datasets:
+            assert callable(dataset)
+            dataset_name = dataset.__name__
+            if dataset_name not in method_map:
+                raise ValueError(f"Dataset method {dataset_name} doesn't "
+                                 "exist. Please check if the passed dataset "
+                                 "is a subset of the following dataset "
+                                 f"methods: {list(method_map.keys())}")
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
-        for dataset in datasets:
-            assert callable(dataset)
-            dataset_name = dataset.__name__
-            if dataset_name not in method_map:
-                raise ValueError(f"Dataset method {dataset_name} doesn't "
-                                 "exist. Please check if the passed dataset "
-                                 "is a subset of the following dataset "
-                                 f"methods: {list(method_map.keys())}")
-
             data_files = method_map[dataset_name]
             data_filepaths = [os.path.join(cache_dir, file)
                               for file in data_files]
```