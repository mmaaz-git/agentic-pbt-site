# Bug Report: pandas.compat._optional.import_optional_dependency Returns None Instead of Module

**Target**: `pandas.compat._optional.import_optional_dependency`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

When `errors='ignore'` is used with an outdated module version, `import_optional_dependency` returns `None` instead of returning the module as documented.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import importlib
import pandas.compat._optional as optional

@given(st.text(min_size=1))
def test_import_optional_dependency_ignore_returns_module(min_version):
    numpy = importlib.import_module('numpy')
    original = numpy.__version__
    try:
        numpy.__version__ = '0.0.1'
        result = optional.import_optional_dependency('numpy', errors='ignore', min_version=min_version)
        assert result is not None
    finally:
        numpy.__version__ = original
```

**Failing input**: `min_version='1.0.0'` (any version newer than '0.0.1')

## Reproducing the Bug

```python
import importlib
import pandas.compat._optional as optional

numpy = importlib.import_module('numpy')
original = numpy.__version__

numpy.__version__ = '0.0.1'
result = optional.import_optional_dependency('numpy', errors='ignore', min_version='999.0.0')

print(f"Result: {result}")
print(f"Expected: <module 'numpy'>")

numpy.__version__ = original
```

## Why This Is A Bug

The docstring explicitly states that when `errors='ignore'`:
> "If the module is not installed, return None, otherwise, **return the module, even if the version is too old.**"

However, the implementation returns `None` when the version is too old, violating the documented contract.

## Fix

```diff
--- a/pandas/compat/_optional.py
+++ b/pandas/compat/_optional.py
@@ -80,7 +80,7 @@ def import_optional_dependency(
             elif errors == "raise":
                 raise ImportError(msg)
             else:
-                return None
+                return module

     return module
```