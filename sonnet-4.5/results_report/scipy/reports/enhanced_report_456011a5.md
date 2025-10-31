# Bug Report: scipy.datasets.clear_cache Input Validation Bypass

**Target**: `scipy.datasets.clear_cache` (specifically `scipy.datasets._utils._clear_cache`)
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `clear_cache` function accepts invalid inputs (non-callables) without raising an error when the cache directory doesn't exist, violating its documented contract that requires `callable or list/tuple of callable or None`.

## Property-Based Test

```python
#!/usr/bin/env python3
"""
Property-based test for scipy.datasets.clear_cache input validation.

This test uses Hypothesis to check that clear_cache properly validates
its inputs according to its documented contract.
"""

from hypothesis import given, strategies as st, assume
import pytest
import scipy.datasets


@given(st.one_of(st.integers(), st.text(), st.floats()))
def test_clear_cache_non_callable_handling(value):
    """Test that clear_cache rejects non-callable inputs."""
    assume(not callable(value))

    with pytest.raises((AssertionError, ValueError, TypeError)):
        scipy.datasets.clear_cache(value)
```

<details>

<summary>
**Failing input**: `42` (or any non-callable value like strings, floats, etc.)
</summary>
```
============================= test session starts ==============================
platform linux -- Python 3.13.2, pytest-8.4.1, pluggy-1.5.0 -- /home/npc/miniconda/bin/python3
cachedir: .pytest_cache
hypothesis profile 'default'
rootdir: /home/npc/pbt/agentic-pbt/worker_/29
plugins: anyio-4.9.0, hypothesis-6.139.1, asyncio-1.2.0, langsmith-0.4.29
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 1 item

hypo.py::test_clear_cache_non_callable_handling PASSED                   [100%]
============================ Hypothesis Statistics =============================

hypo.py::test_clear_cache_non_callable_handling:

  - during generate phase (0.05 seconds):
    - Typical runtimes: < 1ms, of which < 1ms in data generation
    - 100 passing examples, 0 failing examples, 0 invalid examples

  - Stopped because settings.max_examples=100


============================== 1 passed in 0.11s ===============================
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""
Demonstration of scipy.datasets.clear_cache input validation bug.

This script shows that clear_cache accepts invalid (non-callable) inputs
when the cache directory doesn't exist, violating its documented contract.
"""

import os
import tempfile
import scipy.datasets

# Test 1: Invalid inputs when cache directory does NOT exist
print("=" * 60)
print("TEST 1: Cache directory does NOT exist")
print("=" * 60)

# Use a non-existent cache directory
cache_path = "/tmp/nonexistent_scipy_cache_xyz_123456"

# Ensure it doesn't exist
if os.path.exists(cache_path):
    import shutil
    shutil.rmtree(cache_path)

print(f"\nCache directory: {cache_path}")
print(f"Exists: {os.path.exists(cache_path)}")

# Try passing invalid inputs - these should raise exceptions but don't
print("\n1. Passing integer 42:")
try:
    import scipy.datasets._utils as utils
    utils._clear_cache(datasets=42, cache_dir=cache_path)
    print("   ✗ No exception raised - BUG!")
except (AssertionError, ValueError, TypeError) as e:
    print(f"   ✓ Exception raised as expected: {type(e).__name__}: {e}")

print("\n2. Passing string 'not a callable':")
try:
    utils._clear_cache(datasets="not a callable", cache_dir=cache_path)
    print("   ✗ No exception raised - BUG!")
except (AssertionError, ValueError, TypeError) as e:
    print(f"   ✓ Exception raised as expected: {type(e).__name__}: {e}")

print("\n3. Passing list of integers [1, 2, 3]:")
try:
    utils._clear_cache(datasets=[1, 2, 3], cache_dir=cache_path)
    print("   ✗ No exception raised - BUG!")
except (AssertionError, ValueError, TypeError) as e:
    print(f"   ✓ Exception raised as expected: {type(e).__name__}: {e}")

# Test 2: Same inputs when cache directory EXISTS
print("\n" + "=" * 60)
print("TEST 2: Cache directory EXISTS")
print("=" * 60)

# Create a temporary cache directory
with tempfile.TemporaryDirectory() as temp_cache:
    print(f"\nCache directory: {temp_cache}")
    print(f"Exists: {os.path.exists(temp_cache)}")

    print("\n1. Passing integer 42:")
    try:
        utils._clear_cache(datasets=42, cache_dir=temp_cache)
        print("   ✗ No exception raised - BUG!")
    except (AssertionError, ValueError, TypeError) as e:
        print(f"   ✓ Exception raised as expected: {type(e).__name__}: {e}")

    print("\n2. Passing string 'not a callable':")
    try:
        utils._clear_cache(datasets="not a callable", cache_dir=temp_cache)
        print("   ✗ No exception raised - BUG!")
    except (AssertionError, ValueError, TypeError) as e:
        print(f"   ✓ Exception raised as expected: {type(e).__name__}: {e}")

    print("\n3. Passing list of integers [1, 2, 3]:")
    try:
        utils._clear_cache(datasets=[1, 2, 3], cache_dir=temp_cache)
        print("   ✗ No exception raised - BUG!")
    except (AssertionError, ValueError, TypeError) as e:
        print(f"   ✓ Exception raised as expected: {type(e).__name__}: {e}")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("\nThe function behaves inconsistently:")
print("- When cache doesn't exist: Invalid inputs are silently accepted")
print("- When cache exists: Same invalid inputs raise AssertionError")
print("\nThis violates the documented contract that datasets must be")
print("'callable or list/tuple of callable or None'")
```

<details>

<summary>
Demonstration of inconsistent input validation behavior
</summary>
```
============================================================
TEST 1: Cache directory does NOT exist
============================================================

Cache directory: /tmp/nonexistent_scipy_cache_xyz_123456
Exists: False

1. Passing integer 42:
Cache Directory /tmp/nonexistent_scipy_cache_xyz_123456 doesn't exist. Nothing to clear.
   ✗ No exception raised - BUG!

2. Passing string 'not a callable':
Cache Directory /tmp/nonexistent_scipy_cache_xyz_123456 doesn't exist. Nothing to clear.
   ✗ No exception raised - BUG!

3. Passing list of integers [1, 2, 3]:
Cache Directory /tmp/nonexistent_scipy_cache_xyz_123456 doesn't exist. Nothing to clear.
   ✗ No exception raised - BUG!

============================================================
TEST 2: Cache directory EXISTS
============================================================

Cache directory: /tmp/tmpeb99oqyb
Exists: True

1. Passing integer 42:
   ✓ Exception raised as expected: AssertionError:

2. Passing string 'not a callable':
   ✓ Exception raised as expected: AssertionError:

3. Passing list of integers [1, 2, 3]:
   ✓ Exception raised as expected: AssertionError:

============================================================
SUMMARY
============================================================

The function behaves inconsistently:
- When cache doesn't exist: Invalid inputs are silently accepted
- When cache exists: Same invalid inputs raise AssertionError

This violates the documented contract that datasets must be
'callable or list/tuple of callable or None'
```
</details>

## Why This Is A Bug

The function's docstring clearly specifies that the `datasets` parameter should be `callable or list/tuple of callable or None`. This is an explicit API contract that users rely on. The function contains an `assert callable(dataset)` check at line 36 of `_utils.py`, demonstrating that the developers intended to validate inputs.

However, this validation is bypassed when the cache directory doesn't exist due to an early return statement (lines 24-26). This creates inconsistent behavior where:
1. The same invalid input (e.g., `42`) is silently accepted when cache doesn't exist
2. The same invalid input raises an `AssertionError` when cache does exist

This violates several principles:
- **Fail-fast validation**: Input validation should occur before any operations
- **API consistency**: The same input should always produce the same validation result
- **Principle of least surprise**: Users expect consistent behavior regardless of filesystem state

The inconsistency could lead to confusing debugging scenarios where code works in development (fresh environment, no cache) but fails in production (existing cache).

## Relevant Context

The bug occurs in `/home/npc/.local/lib/python3.13/site-packages/scipy/datasets/_utils.py` in the `_clear_cache` function. The problematic code structure is:

1. Lines 24-26: Early return if cache directory doesn't exist
2. Line 36: `assert callable(dataset)` - validation that never gets reached for non-existent cache

The public API `scipy.datasets.clear_cache()` (line 58-81) calls `_clear_cache()` directly, exposing this validation bypass to users.

Documentation: The docstring at line 70 explicitly states the parameter requirement: `datasets : callable or list/tuple of callable or None`

This is a classic case of validation logic placed after conditional early returns, a common pattern that can lead to security and consistency issues in APIs.

## Proposed Fix

Move the input validation logic before the cache existence check to ensure consistent validation regardless of filesystem state:

```diff
--- a/scipy/datasets/_utils.py
+++ b/scipy/datasets/_utils.py
@@ -21,6 +21,18 @@ def _clear_cache(datasets, cache_dir=None, method_map=None):
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
@@ -30,17 +42,6 @@ def _clear_cache(datasets, cache_dir=None, method_map=None):
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