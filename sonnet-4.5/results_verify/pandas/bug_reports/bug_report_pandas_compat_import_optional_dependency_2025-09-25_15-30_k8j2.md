# Bug Report: import_optional_dependency raises with errors='ignore'

**Target**: `pandas.compat._optional.import_optional_dependency`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`import_optional_dependency` raises `ImportError` even when `errors='ignore'` if a module lacks a `__version__` attribute and `min_version` is specified.

## Property-Based Test

```python
import sys
import types

import pytest
from hypothesis import given, settings, strategies as st

from pandas.compat._optional import import_optional_dependency


def test_import_optional_dependency_ignore_with_no_version_attribute():
    mock_module = types.ModuleType("test_no_version")
    sys.modules["test_no_version"] = mock_module

    try:
        result = import_optional_dependency(
            "test_no_version",
            errors="ignore",
            min_version="1.0.0"
        )
    except ImportError:
        pytest.fail("errors='ignore' should not raise ImportError for missing __version__")
    finally:
        if "test_no_version" in sys.modules:
            del sys.modules["test_no_version"]
```

**Failing input**: Module without `__version__` attribute, with `errors='ignore'` and `min_version='1.0.0'`

## Reproducing the Bug

```python
import sys
import types

from pandas.compat._optional import import_optional_dependency

mock_module = types.ModuleType("test_no_version")
sys.modules["test_no_version"] = mock_module

try:
    result = import_optional_dependency(
        "test_no_version",
        errors="ignore",
        min_version="1.0.0"
    )
    print(f"Success: {result}")
except ImportError as e:
    print(f"Bug: ImportError raised with errors='ignore': {e}")
finally:
    if "test_no_version" in sys.modules:
        del sys.modules["test_no_version"]
```

## Why This Is A Bug

The docstring states: `"ignore: If the module is not installed, return None, otherwise, return the module, even if the version is too old."` When `errors='ignore'`, the function should never raise, but it does when `get_version()` raises `ImportError` for missing `__version__` attribute.

## Fix

```diff
--- a/pandas/compat/_optional.py
+++ b/pandas/compat/_optional.py
@@ -147,7 +147,11 @@ def import_optional_dependency(
     else:
         module_to_get = module
     minimum_version = min_version if min_version is not None else VERSIONS.get(parent)
     if minimum_version:
-        version = get_version(module_to_get)
+        try:
+            version = get_version(module_to_get)
+        except ImportError:
+            if errors == "raise":
+                raise
+            return None if errors == "warn" else module
         if version and Version(version) < Version(minimum_version):
             msg = (
                 f"Pandas requires version '{minimum_version}' or newer of '{parent}' "
```