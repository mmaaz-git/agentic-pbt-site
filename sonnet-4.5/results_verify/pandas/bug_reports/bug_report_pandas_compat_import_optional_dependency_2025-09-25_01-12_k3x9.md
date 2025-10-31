# Bug Report: pandas.compat import_optional_dependency errors='ignore' Contract Violation

**Target**: `pandas.compat._optional.import_optional_dependency`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`import_optional_dependency` with `errors='ignore'` violates its documented contract by returning `None` when a module's version is too old, and by raising `ImportError` when a module lacks a `__version__` attribute. The docstring explicitly states that with `errors='ignore'`, the function should "return the module, even if the version is too old."

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pandas.compat._optional import import_optional_dependency

def test_errors_ignore_returns_module_when_old_version():
    result = import_optional_dependency("hypothesis", errors="ignore", min_version="999.0.0")
    assert result is not None, "errors='ignore' should return module even when version is too old"

def test_errors_ignore_module_without_version():
    result = import_optional_dependency("sys", errors="ignore", min_version="1.0.0")
    assert result is not None, "errors='ignore' should not raise even if module has no __version__"
```

**Failing input**: `name="hypothesis", errors="ignore", min_version="999.0.0"`
**Second failing input**: `name="sys", errors="ignore", min_version="1.0.0"`

## Reproducing the Bug

```python
from pandas.compat._optional import import_optional_dependency

result = import_optional_dependency("hypothesis", errors="ignore", min_version="999.0.0")
print(f"Result: {result}")

try:
    result2 = import_optional_dependency("sys", errors="ignore", min_version="1.0.0")
except ImportError as e:
    print(f"Raised ImportError: {e}")
```

## Why This Is A Bug

The function's docstring at line 112-113 states:
> `ignore`: If the module is not installed, return None, otherwise, return the module, even if the version is too old.

However, the implementation at lines 163-166 returns `None` when the version is too old and `errors='ignore'`:

```python
elif errors == "raise":
    raise ImportError(msg)
else:
    return None  # BUG: should return module when errors='ignore'
```

Additionally, when a module doesn't have a `__version__` attribute and `min_version` is specified, `get_version()` at line 150 raises an `ImportError` regardless of the `errors` parameter, violating the "ignore" contract.

## Fix

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
@@ -162,7 +167,7 @@ def import_optional_dependency(
             elif errors == "raise":
                 raise ImportError(msg)
             else:
-                return None
+                return module

     return module
```