# Bug Report: scipy.datasets.clear_cache() Bypasses Input Validation When Cache Directory Doesn't Exist

**Target**: `scipy.datasets._utils._clear_cache()`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `_clear_cache()` function skips validation of dataset callables when the cache directory doesn't exist, allowing invalid datasets to be passed without raising the documented `ValueError`. This creates inconsistent behavior where the same invalid input is sometimes accepted and sometimes rejected based on filesystem state.

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

if __name__ == "__main__":
    test_clear_cache_validates_regardless_of_cache_existence()
```

<details>

<summary>
**Failing input**: `cache_exists=False`
</summary>
```
Cache Directory /tmp/test_cache_False doesn't exist. Nothing to clear.
Cache Directory /tmp/test_cache_False doesn't exist. Nothing to clear.
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/47/hypo.py", line 30, in <module>
    test_clear_cache_validates_regardless_of_cache_existence()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/47/hypo.py", line 8, in test_clear_cache_validates_regardless_of_cache_existence
    def test_clear_cache_validates_regardless_of_cache_existence(cache_exists):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/47/hypo.py", line 23, in test_clear_cache_validates_regardless_of_cache_existence
    with pytest.raises(ValueError, match="Dataset method .* doesn't exist"):
         ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/_pytest/raises.py", line 712, in __exit__
    fail(f"DID NOT RAISE {self.expected_exceptions[0]!r}")
    ~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/_pytest/outcomes.py", line 177, in fail
    raise Failed(msg=reason, pytrace=pytrace)
Failed: DID NOT RAISE <class 'ValueError'>
Falsifying example: test_clear_cache_validates_regardless_of_cache_existence(
    cache_exists=False,
)
```
</details>

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

<details>

<summary>
Invalid dataset accepted when cache doesn't exist
</summary>
```
Cache exists: False
Cache Directory /tmp/scipy_test_nonexistent doesn't exist. Nothing to clear.
BUG: No ValueError raised!
```
</details>

## Why This Is A Bug

This violates expected behavior because the function contains explicit validation logic (lines 36-42 in `_utils.py`) that is intended to validate dataset callables and raise a `ValueError` for invalid ones. The validation includes:

1. An assertion that the dataset is callable (line 36)
2. A check that the dataset name exists in the method_map (line 38)
3. A detailed error message showing valid dataset methods: `['ascent', 'electrocardiogram', 'face']` (lines 39-42)

However, an early return at line 26 when the cache directory doesn't exist bypasses this validation entirely. This creates inconsistent behavior where:
- Invalid datasets are **rejected** when the cache directory exists (correct behavior)
- Invalid datasets are **accepted** when the cache directory doesn't exist (incorrect behavior)

This violates the principle of fail-fast validation - invalid input should always be rejected immediately, regardless of filesystem state. The inconsistency makes the API unpredictable and harder to test reliably.

## Relevant Context

The `_clear_cache` function is the internal implementation called by the public `scipy.datasets.clear_cache()` API. Valid dataset methods are defined in `_registry.py` as:
- `'ascent'`: Maps to file `ascent.dat`
- `'electrocardiogram'`: Maps to file `ecg.dat`
- `'face'`: Maps to file `face.dat`

The function is located in: `/scipy/datasets/_utils.py`

The early return causing the bug is at lines 24-26:
```python
if not os.path.exists(cache_dir):
    print(f"Cache Directory {cache_dir} doesn't exist. Nothing to clear.")
    return
```

This occurs before the validation logic at lines 36-42, which is never reached when the cache directory doesn't exist.

## Proposed Fix

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