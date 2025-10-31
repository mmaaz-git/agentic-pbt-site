# Bug Report: scipy.datasets.clear_cache() Skips Input Validation

**Target**: `scipy.datasets.clear_cache()`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `clear_cache()` function skips input validation when the cache directory doesn't exist, allowing invalid dataset callables to be passed without raising the documented `ValueError`.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pytest
import os
import shutil
from scipy.datasets._utils import _clear_cache

@given(st.booleans())
def test_clear_cache_validates_regardless_of_cache_existence(cache_exists):
    def invalid_dataset():
        pass

    invalid_dataset.__name__ = "not_a_real_dataset"

    cache_dir = f"/tmp/test_cache_{cache_exists}"

    if cache_exists:
        os.makedirs(cache_dir, exist_ok=True)
    else:
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)

    try:
        with pytest.raises(ValueError, match="Dataset method .* doesn't exist"):
            _clear_cache([invalid_dataset], cache_dir=cache_dir)
    finally:
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
```

**Failing input**: `cache_exists=False` (when cache directory doesn't exist)

## Reproducing the Bug

```python
import os
import shutil
from scipy.datasets._utils import _clear_cache

def invalid_dataset():
    pass

invalid_dataset.__name__ = "not_a_dataset"

cache_dir = "/tmp/scipy_test_nonexistent"
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)

print(f"Cache exists: {os.path.exists(cache_dir)}")

try:
    _clear_cache([invalid_dataset], cache_dir=cache_dir)
    print("BUG: No ValueError raised!")
except ValueError as e:
    print(f"Correctly raised ValueError: {e}")
```

Output:
```
Cache exists: False
Cache Directory /tmp/scipy_test_nonexistent doesn't exist. Nothing to clear.
BUG: No ValueError raised!
```

## Why This Is A Bug

The function's contract (lines 36-42 in `_utils.py`) promises to validate that callables are valid dataset methods and raise `ValueError` for invalid ones. However, the early return on line 26 when the cache directory doesn't exist bypasses this validation entirely.

This violates the principle of fail-fast validation and creates inconsistent behavior where the same invalid input is sometimes accepted and sometimes rejected depending on filesystem state.

## Fix

```diff
--- a/scipy/datasets/_utils.py
+++ b/scipy/datasets/_utils.py
@@ -21,11 +21,6 @@ def _clear_cache(datasets, cache_dir=None, method_map=None):
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
@@ -44,6 +39,11 @@ def _clear_cache(datasets, cache_dir=None, method_map=None):
             data_files = method_map[dataset_name]
             data_filepaths = [os.path.join(cache_dir, file)
                               for file in data_files]
+
+            if not os.path.exists(cache_dir):
+                print(f"Cache Directory {cache_dir} doesn't exist. Nothing to clear.")
+                return
+
             for data_filepath in data_filepaths:
                 if os.path.exists(data_filepath):
                     print("Cleaning the file "
```

This ensures validation happens before the early return, maintaining the function's contract regardless of cache directory existence.