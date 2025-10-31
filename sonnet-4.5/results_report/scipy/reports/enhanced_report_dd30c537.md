# Bug Report: scipy.datasets.clear_cache Bypasses Input Validation When Cache Directory Absent

**Target**: `scipy.datasets.clear_cache`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `scipy.datasets.clear_cache()` function silently accepts invalid input types (strings, integers, dictionaries) when the cache directory doesn't exist, but correctly rejects these same inputs when the directory does exist, creating inconsistent validation behavior.

## Property-Based Test

```python
#!/usr/bin/env python3
"""
Hypothesis-based property test for scipy.datasets.clear_cache
This test demonstrates that the function silently accepts invalid inputs
when the cache directory doesn't exist.
"""

from hypothesis import given, strategies as st, settings
import pytest
import scipy.datasets
import os
import shutil

# Ensure cache directory doesn't exist for consistent test conditions
cache_dir = os.path.expanduser("~/.cache/scipy-data")
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)
    print(f"Removed existing cache directory: {cache_dir}")

print(f"Cache directory exists: {os.path.exists(cache_dir)}")

@given(st.text(min_size=1))
@settings(max_examples=5)
def test_clear_cache_rejects_arbitrary_strings(text_input):
    """Property: clear_cache should reject any string input"""
    if callable(text_input):
        return

    with pytest.raises((ValueError, TypeError, AssertionError)):
        scipy.datasets.clear_cache(text_input)

@given(st.integers())
@settings(max_examples=5)
def test_clear_cache_rejects_integers(int_input):
    """Property: clear_cache should reject integer inputs"""
    with pytest.raises((ValueError, TypeError, AssertionError)):
        scipy.datasets.clear_cache(int_input)

@given(st.dictionaries(st.text(), st.text()))
@settings(max_examples=5)
def test_clear_cache_rejects_dictionaries(dict_input):
    """Property: clear_cache should reject dictionary inputs"""
    with pytest.raises((ValueError, TypeError, AssertionError)):
        scipy.datasets.clear_cache(dict_input)

if __name__ == "__main__":
    # Run the tests using pytest
    import sys
    import subprocess

    # Save this file's name
    test_file = __file__

    # Run pytest on this file
    result = subprocess.run(
        [sys.executable, "-m", "pytest", test_file, "-v", "--tb=short", "--hypothesis-show-statistics"],
        capture_output=True,
        text=True
    )

    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    print(f"\nReturn code: {result.returncode}")
    if result.returncode != 0:
        print("Tests FAILED - This confirms the bug!")
```

<details>

<summary>
**Failing input**: `'0'` (string), `0` (integer), `{}` (dictionary)
</summary>
```
Cache directory exists: False
============================= test session starts ==============================
platform linux -- Python 3.13.2, pytest-8.4.1, pluggy-1.5.0 -- /home/npc/miniconda/bin/python3
cachedir: .pytest_cache
hypothesis profile 'default'
rootdir: /home/npc/pbt/agentic-pbt/worker_/48
plugins: anyio-4.9.0, hypothesis-6.139.1, asyncio-1.2.0, langsmith-0.4.29
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 3 items

hypo.py::test_clear_cache_rejects_arbitrary_strings FAILED               [ 33%]
hypo.py::test_clear_cache_rejects_integers FAILED                        [ 66%]
hypo.py::test_clear_cache_rejects_dictionaries FAILED                    [100%]

=================================== FAILURES ===================================
__________________ test_clear_cache_rejects_arbitrary_strings __________________
hypo.py:23: in test_clear_cache_rejects_arbitrary_strings
    @settings(max_examples=5)
                   ^^^
hypo.py:29: in test_clear_cache_rejects_arbitrary_strings
    with pytest.raises((ValueError, TypeError, AssertionError)):
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E   Failed: DID NOT RAISE any of (<class 'ValueError'>, <class 'TypeError'>, <class 'AssertionError'>)
E   Falsifying example: test_clear_cache_rejects_arbitrary_strings(
E       text_input='0',  # or any other generated value
E   )
----------------------------- Captured stdout call -----------------------------
Cache Directory /home/npc/.cache/scipy-data doesn't exist. Nothing to clear.
[... repeated many times ...]
______________________ test_clear_cache_rejects_integers _______________________
hypo.py:33: in test_clear_cache_rejects_integers
    @settings(max_examples=5)
                   ^^^
hypo.py:36: in test_clear_cache_rejects_integers
    with pytest.raises((ValueError, TypeError, AssertionError)):
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E   Failed: DID NOT RAISE any of (<class 'ValueError'>, <class 'TypeError'>, <class 'AssertionError'>)
E   Falsifying example: test_clear_cache_rejects_integers(
E       int_input=0,  # or any other generated value
E   )
[... similar output ...]
____________________ test_clear_cache_rejects_dictionaries _____________________
hypo.py:40: in test_clear_cache_rejects_dictionaries
    @settings(max_examples=5)
                   ^^^
hypo.py:43: in test_clear_cache_rejects_dictionaries
    with pytest.raises((ValueError, TypeError, AssertionError)):
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E   Failed: DID NOT RAISE any of (<class 'ValueError'>, <class 'TypeError'>, <class 'AssertionError'>)
E   Falsifying example: test_clear_cache_rejects_dictionaries(
E       dict_input={},  # or any other generated value
E   )
[... similar output ...]
============================== 3 failed in 1.50s ===============================

Return code: 1
Tests FAILED - This confirms the bug!
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""
Minimal reproduction of scipy.datasets.clear_cache bug.
This demonstrates that invalid inputs are silently accepted when cache doesn't exist.
"""

import scipy.datasets
import os
import shutil

# Ensure cache directory doesn't exist for consistent reproduction
cache_dir = os.path.expanduser("~/.cache/scipy-data")
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)
    print(f"Removed existing cache directory: {cache_dir}")

print("Testing with cache directory NOT existing:")
print("-" * 50)

# Test with invalid string input
print("\n1. Testing with string input 'invalid_string':")
try:
    scipy.datasets.clear_cache("invalid_string")
    print("   No error raised - BUG!")
except (ValueError, TypeError, AssertionError) as e:
    print(f"   Error raised as expected: {e}")

# Test with invalid integer input
print("\n2. Testing with integer input 42:")
try:
    scipy.datasets.clear_cache(42)
    print("   No error raised - BUG!")
except (ValueError, TypeError, AssertionError) as e:
    print(f"   Error raised as expected: {e}")

# Test with invalid dict input
print("\n3. Testing with dict input {'key': 'value'}:")
try:
    scipy.datasets.clear_cache({"key": "value"})
    print("   No error raised - BUG!")
except (ValueError, TypeError, AssertionError) as e:
    print(f"   Error raised as expected: {e}")

print("\n" + "=" * 50)
print("\nNow creating cache directory and testing again...")
print("=" * 50)

# Create the cache directory
os.makedirs(cache_dir, exist_ok=True)
print(f"Created cache directory: {cache_dir}")

print("\nTesting with cache directory EXISTING:")
print("-" * 50)

# Test with invalid string input (should fail now)
print("\n1. Testing with string input 'invalid_string':")
try:
    scipy.datasets.clear_cache("invalid_string")
    print("   No error raised - BUG!")
except (ValueError, TypeError, AssertionError) as e:
    print(f"   Error raised as expected: {type(e).__name__}: {e}")

# Test with invalid integer input (should fail now)
print("\n2. Testing with integer input 42:")
try:
    scipy.datasets.clear_cache(42)
    print("   No error raised - BUG!")
except (ValueError, TypeError, AssertionError) as e:
    print(f"   Error raised as expected: {type(e).__name__}: {e}")

# Test with invalid dict input (should fail now)
print("\n3. Testing with dict input {'key': 'value'}:")
try:
    scipy.datasets.clear_cache({"key": "value"})
    print("   No error raised - BUG!")
except (ValueError, TypeError, AssertionError) as e:
    print(f"   Error raised as expected: {type(e).__name__}: {e}")

# Clean up
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)
    print(f"\nCleaned up cache directory: {cache_dir}")
```

<details>

<summary>
Inconsistent validation behavior based on cache directory existence
</summary>
```
Removed existing cache directory: /home/npc/.cache/scipy-data
Testing with cache directory NOT existing:
--------------------------------------------------

1. Testing with string input 'invalid_string':
Cache Directory /home/npc/.cache/scipy-data doesn't exist. Nothing to clear.
   No error raised - BUG!

2. Testing with integer input 42:
Cache Directory /home/npc/.cache/scipy-data doesn't exist. Nothing to clear.
   No error raised - BUG!

3. Testing with dict input {'key': 'value'}:
Cache Directory /home/npc/.cache/scipy-data doesn't exist. Nothing to clear.
   No error raised - BUG!

==================================================

Now creating cache directory and testing again...
==================================================
Created cache directory: /home/npc/.cache/scipy-data

Testing with cache directory EXISTING:
--------------------------------------------------

1. Testing with string input 'invalid_string':
   Error raised as expected: AssertionError:

2. Testing with integer input 42:
   Error raised as expected: AssertionError:

3. Testing with dict input {'key': 'value'}:
   Error raised as expected: AssertionError:

Cleaned up cache directory: /home/npc/.cache/scipy-data
```
</details>

## Why This Is A Bug

This violates the function's documented API contract. According to the docstring at `/home/npc/.local/lib/python3.13/site-packages/scipy/datasets/_utils.py:70`:

```
Parameters
----------
datasets : callable or list/tuple of callable or None
```

The function explicitly states it should only accept:
1. `None` (to clear all cached data)
2. A callable (e.g., `scipy.datasets.ascent`)
3. A list/tuple of callables

The implementation contains validation logic at line 36 (`assert callable(dataset)`) that demonstrates the developers' intent to enforce these requirements. However, the code structure causes this validation to be bypassed:

- **Lines 24-26**: The function checks if `cache_dir` exists and returns early if not
- **Line 36**: Input validation occurs AFTER the early return

This creates two distinct behaviors:
- **Cache directory doesn't exist**: Invalid inputs silently accepted (returns after printing message)
- **Cache directory exists**: Invalid inputs correctly rejected with `AssertionError`

This inconsistency violates the principle of fail-fast error handling - API contracts should be enforced consistently regardless of filesystem state. User errors should be caught immediately, not conditionally based on whether a cache directory exists.

## Relevant Context

The bug affects new installations or first-time users most severely, as they won't have a cache directory yet. CI/CD environments that start with clean filesystems will also encounter this inconsistent behavior.

The issue is in `/home/npc/.local/lib/python3.13/site-packages/scipy/datasets/_utils.py` in the `_clear_cache` internal function. The public `clear_cache` function at line 58 simply delegates to `_clear_cache`.

While no actual harm occurs (since there's nothing to clear when the directory doesn't exist), the inconsistent validation can mask programming errors and make debugging more difficult for users who may be passing incorrect arguments.

## Proposed Fix

```diff
--- a/scipy/datasets/_utils.py
+++ b/scipy/datasets/_utils.py
@@ -19,12 +19,25 @@ def _clear_cache(datasets, cache_dir=None, method_map=None):
             raise ImportError("Missing optional dependency 'pooch' required "
                               "for scipy.datasets module. Please use pip or "
                               "conda to install 'pooch'.")
         cache_dir = platformdirs.user_cache_dir("scipy-data")

+    # Validate input parameters before checking cache directory existence
+    if datasets is not None:
+        if not isinstance(datasets, list | tuple):
+            # single dataset method passed should be converted to list
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
-            # single dataset method passed should be converted to list
-            datasets = [datasets, ]
         for dataset in datasets:
-            assert callable(dataset)
             dataset_name = dataset.__name__  # Name of the dataset method
```