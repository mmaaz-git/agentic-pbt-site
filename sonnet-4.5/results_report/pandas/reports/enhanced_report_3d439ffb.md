# Bug Report: pandas.compat.import_optional_dependency errors='ignore' Contract Violation

**Target**: `pandas.compat._optional.import_optional_dependency`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `import_optional_dependency` function with `errors='ignore'` violates its documented contract by returning `None` instead of the module when a module's version is too old, and by raising `ImportError` when a module lacks a `__version__` attribute.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pandas.compat._optional import import_optional_dependency

def test_errors_ignore_returns_module_when_old_version():
    """Test that errors='ignore' returns the module even when version is too old.

    According to the docstring, when errors='ignore':
    'return the module, even if the version is too old.'
    """
    result = import_optional_dependency("hypothesis", errors="ignore", min_version="999.0.0")
    assert result is not None, "errors='ignore' should return module even when version is too old"
    print(f"Test 1 FAILED: Expected module, got {result}")

def test_errors_ignore_module_without_version():
    """Test that errors='ignore' handles modules without __version__ gracefully.

    According to the contract, errors='ignore' should not raise exceptions.
    """
    try:
        result = import_optional_dependency("sys", errors="ignore", min_version="1.0.0")
        assert result is not None, "errors='ignore' should not raise even if module has no __version__"
        print(f"Test 2 passed: Got result {result}")
    except ImportError as e:
        print(f"Test 2 FAILED: errors='ignore' raised ImportError: {e}")

# Run the tests
print("Running property-based tests for import_optional_dependency with errors='ignore':")
print("-" * 70)
test_errors_ignore_returns_module_when_old_version()
test_errors_ignore_module_without_version()
```

<details>

<summary>
**Failing input**: `name="hypothesis", errors="ignore", min_version="999.0.0"` and `name="sys", errors="ignore", min_version="1.0.0"`
</summary>
```
Running property-based tests for import_optional_dependency with errors='ignore':
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/57/hypo.py", line 29, in <module>
    test_errors_ignore_returns_module_when_old_version()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/57/hypo.py", line 11, in test_errors_ignore_returns_module_when_old_version
    assert result is not None, "errors='ignore' should return module even when version is too old"
           ^^^^^^^^^^^^^^^^^^
AssertionError: errors='ignore' should return module even when version is too old
```
</details>

## Reproducing the Bug

```python
from pandas.compat._optional import import_optional_dependency

# Test Case 1: Module with old version (hypothesis exists but version is set impossibly high)
print("Test Case 1: Module with old version")
print("Calling: import_optional_dependency('hypothesis', errors='ignore', min_version='999.0.0')")
result = import_optional_dependency("hypothesis", errors="ignore", min_version="999.0.0")
print(f"Result: {result}")
print(f"Result type: {type(result)}")
print()

# Test Case 2: Module without __version__ attribute (sys has no __version__)
print("Test Case 2: Module without __version__ attribute")
print("Calling: import_optional_dependency('sys', errors='ignore', min_version='1.0.0')")
try:
    result2 = import_optional_dependency("sys", errors="ignore", min_version="1.0.0")
    print(f"Result: {result2}")
    print(f"Result type: {type(result2)}")
except ImportError as e:
    print(f"Raised ImportError: {e}")
    print(f"Exception type: {type(e).__name__}")
```

<details>

<summary>
Output shows None returned for old version module and ImportError raised for module without __version__
</summary>
```
Test Case 1: Module with old version
Calling: import_optional_dependency('hypothesis', errors='ignore', min_version='999.0.0')
Result: None
Result type: <class 'NoneType'>

Test Case 2: Module without __version__ attribute
Calling: import_optional_dependency('sys', errors='ignore', min_version='1.0.0')
Raised ImportError: Can't determine version for sys
Exception type: ImportError
```
</details>

## Why This Is A Bug

The function's docstring at lines 110-113 explicitly states:
> "ignore: If the module is not installed, return None, otherwise, return the module, even if the version is too old. It's expected that users validate the version locally when using `errors="ignore"`"

However, the implementation violates this contract in two ways:

1. **Bug #1 (line 166)**: When a module's version is too old and `errors='ignore'`, the function returns `None` instead of returning the module. The code at line 166 returns `None` in the else clause after checking the version, which directly contradicts the documented behavior.

2. **Bug #2 (line 150)**: When a module doesn't have a `__version__` attribute and `min_version` is specified, the `get_version()` function called at line 150 raises an `ImportError` (from line 78 of `get_version()`), regardless of the `errors` parameter value. This violates the "ignore" contract which should not raise exceptions.

## Relevant Context

The `errors='ignore'` mode is designed to allow users to handle version validation themselves, as mentioned in the docstring comment referencing `io/html.py`. This is useful for scenarios where:
- Users want to implement custom version checking logic
- Graceful degradation is preferred over hard failures
- Built-in modules like `sys` that don't have `__version__` attributes need to be imported

The current implementation makes `errors='ignore'` behave inconsistently with its documented purpose, essentially making it behave like `errors='warn'` for version mismatches (returning None) while still raising exceptions for modules without version attributes.

Code location: `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/compat/_optional.py`

## Proposed Fix

```diff
--- a/pandas/compat/_optional.py
+++ b/pandas/compat/_optional.py
@@ -147,7 +147,12 @@ def import_optional_dependency(
         module_to_get = module
     minimum_version = min_version if min_version is not None else VERSIONS.get(parent)
     if minimum_version:
-        version = get_version(module_to_get)
+        try:
+            version = get_version(module_to_get)
+        except ImportError:
+            if errors == "raise":
+                raise
+            version = None
         if version and Version(version) < Version(minimum_version):
             msg = (
                 f"Pandas requires version '{minimum_version}' or newer of '{parent}' "
@@ -163,7 +168,7 @@ def import_optional_dependency(
             elif errors == "raise":
                 raise ImportError(msg)
             else:
-                return None
+                return module

     return module
```