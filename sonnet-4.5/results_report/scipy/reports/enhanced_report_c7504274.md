# Bug Report: scipy.datasets.clear_cache Input Validation Bypass When Cache Directory Missing

**Target**: `scipy.datasets.clear_cache`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`scipy.datasets.clear_cache()` bypasses input validation when the cache directory doesn't exist, silently accepting invalid inputs that would normally raise exceptions.

## Property-Based Test

```python
import pytest
from hypothesis import given, strategies as st
import scipy.datasets
import os
import shutil
import platformdirs

# Ensure cache directory doesn't exist for proper testing
cache_dir = platformdirs.user_cache_dir("scipy-data")
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)


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


if __name__ == "__main__":
    # Run the tests
    print("Running property-based test for non-callables...")
    try:
        test_clear_cache_rejects_non_callables()
    except AssertionError as e:
        print(f"Property test failed with AssertionError")
        print(f"Falsifying example: test_clear_cache_rejects_non_callables(invalid_input='')")
        print(f"The test expects an exception but none was raised")

    print("\nRunning test for invalid callables...")
    try:
        test_clear_cache_rejects_invalid_callables()
        print("Test passed")
    except AssertionError as e:
        print(f"Test failed with AssertionError")
        print(f"Expected ValueError but none was raised")
```

<details>

<summary>
**Failing input**: `invalid_input=''` (any non-callable string)
</summary>
```
============================= test session starts ==============================
platform linux -- Python 3.13.2, pytest-8.4.1, pluggy-1.5.0 -- /home/npc/miniconda/bin/python3
cachedir: .pytest_cache
hypothesis profile 'default'
rootdir: /home/npc/pbt/agentic-pbt/worker_/44
plugins: anyio-4.9.0, hypothesis-6.139.1, asyncio-1.2.0, langsmith-0.4.29
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 2 items

hypo.py::test_clear_cache_rejects_non_callables FAILED                   [ 50%]
hypo.py::test_clear_cache_rejects_invalid_callables FAILED               [100%]

=================================== FAILURES ===================================
____________________ test_clear_cache_rejects_non_callables ____________________
hypo.py:15: in test_clear_cache_rejects_non_callables
    def test_clear_cache_rejects_non_callables(invalid_input):
                   ^^^
hypo.py:18: in test_clear_cache_rejects_non_callables
    with pytest.raises((AssertionError, TypeError)):
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E   Failed: DID NOT RAISE any of (<class 'AssertionError'>, <class 'TypeError'>)
E   Falsifying example: test_clear_cache_rejects_non_callables(
E       invalid_input='',  # or any other generated value
E   )
----------------------------- Captured stdout call -----------------------------
Cache Directory /home/npc/.cache/scipy-data doesn't exist. Nothing to clear.
Cache Directory /home/npc/.cache/scipy-data doesn't exist. Nothing to clear.
[... repeated 90+ times ...]
__________________ test_clear_cache_rejects_invalid_callables __________________
hypo.py:27: in test_clear_cache_rejects_invalid_callables
    with pytest.raises(ValueError, match="Dataset method .* doesn't exist"):
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E   Failed: DID NOT RAISE <class 'ValueError'>
----------------------------- Captured stdout call -----------------------------
Cache Directory /home/npc/.cache/scipy-data doesn't exist. Nothing to clear.
=========================== short test summary info ============================
FAILED hypo.py::test_clear_cache_rejects_non_callables - Failed: DID NOT RAIS...
FAILED hypo.py::test_clear_cache_rejects_invalid_callables - Failed: DID NOT ...
============================== 2 failed in 0.99s ===============================
```
</details>

## Reproducing the Bug

```python
import scipy.datasets
import os
import shutil
import platformdirs

# Ensure cache directory doesn't exist for the first tests
cache_dir = platformdirs.user_cache_dir("scipy-data")
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)

print("=" * 60)
print("Test 1: Passing non-callable string when cache doesn't exist")
print("=" * 60)
try:
    scipy.datasets.clear_cache("not_callable")
    print("No exception raised - function returned successfully")
except (AssertionError, TypeError, ValueError) as e:
    print(f"Exception raised: {type(e).__name__}: {e}")

print("\n" + "=" * 60)
print("Test 2: Passing invalid callable when cache doesn't exist")
print("=" * 60)
def invalid_dataset():
    """This is not a valid scipy dataset function"""
    pass

try:
    scipy.datasets.clear_cache(invalid_dataset)
    print("No exception raised - function returned successfully")
except (AssertionError, TypeError, ValueError) as e:
    print(f"Exception raised: {type(e).__name__}: {e}")

# Now create cache directory to test the same inputs when cache exists
print("\n" + "=" * 60)
print("Creating cache directory and testing same inputs...")
print("=" * 60)
os.makedirs(cache_dir, exist_ok=True)

print("\n" + "=" * 60)
print("Test 3: Passing non-callable string when cache EXISTS")
print("=" * 60)
try:
    scipy.datasets.clear_cache("not_callable")
    print("No exception raised - function returned successfully")
except (AssertionError, TypeError, ValueError) as e:
    print(f"Exception raised: {type(e).__name__}: {e}")

print("\n" + "=" * 60)
print("Test 4: Passing invalid callable when cache EXISTS")
print("=" * 60)
try:
    scipy.datasets.clear_cache(invalid_dataset)
    print("No exception raised - function returned successfully")
except (AssertionError, TypeError, ValueError) as e:
    print(f"Exception raised: {type(e).__name__}: {e}")

# Clean up
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)
```

<details>

<summary>
Inconsistent validation behavior between cache states
</summary>
```
============================================================
Test 1: Passing non-callable string when cache doesn't exist
============================================================
Cache Directory /home/npc/.cache/scipy-data doesn't exist. Nothing to clear.
No exception raised - function returned successfully

============================================================
Test 2: Passing invalid callable when cache doesn't exist
============================================================
Cache Directory /home/npc/.cache/scipy-data doesn't exist. Nothing to clear.
No exception raised - function returned successfully

============================================================
Creating cache directory and testing same inputs...
============================================================

============================================================
Test 3: Passing non-callable string when cache EXISTS
============================================================
Exception raised: AssertionError:

============================================================
Test 4: Passing invalid callable when cache EXISTS
============================================================
Exception raised: ValueError: Dataset method invalid_dataset doesn't exist. Please check if the passed dataset is a subset of the following dataset methods: ['ascent', 'electrocardiogram', 'face']
```
</details>

## Why This Is A Bug

This violates the function's documented contract and creates inconsistent API behavior. The documentation (lines 69-70 in `_utils.py`) explicitly states the parameter must be "callable or list/tuple of callable or None". The function contains validation code:

1. Line 36: `assert callable(dataset)` - validates input is callable
2. Lines 38-42: Raises `ValueError` for invalid dataset names

However, these validations are bypassed due to an early return on lines 24-26 when the cache directory doesn't exist. This means:

- **Same invalid input → Different behavior** depending on whether cache exists
- **Violates Fail Fast principle**: Invalid inputs should be rejected immediately
- **Masks programming errors**: Users won't discover they're passing invalid arguments
- **Inconsistent with documentation**: Documentation specifies valid input types but implementation doesn't enforce them consistently

## Relevant Context

The issue is in `/home/npc/pbt/agentic-pbt/envs/scipy_env/lib/python3.13/site-packages/scipy/datasets/_utils.py`. The `_clear_cache` function checks if the cache directory exists (line 24) and returns early if it doesn't, before reaching the input validation code (lines 36-42).

Valid dataset methods are defined in `_registry.py`:
- `ascent`
- `electrocardiogram`
- `face`

Documentation reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.datasets.clear_cache.html

## Proposed Fix

Move input validation before the cache directory existence check to ensure consistent behavior:

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
@@ -41,6 +36,11 @@ def _clear_cache(datasets, cache_dir=None, method_map=None):
                                  "is a subset of the following dataset "
                                  f"methods: {list(method_map.keys())}")

+        # Now check if cache exists after validation
+        if not os.path.exists(cache_dir):
+            print(f"Cache Directory {cache_dir} doesn't exist. Nothing to clear.")
+            return
+
             data_files = method_map[dataset_name]
             data_filepaths = [os.path.join(cache_dir, file)
                               for file in data_files]
```