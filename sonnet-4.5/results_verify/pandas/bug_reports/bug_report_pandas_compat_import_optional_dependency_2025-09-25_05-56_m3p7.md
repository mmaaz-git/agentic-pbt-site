# Bug Report: pandas.compat import_optional_dependency Contract Violation

**Target**: `pandas.compat._optional.import_optional_dependency`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

When `errors="ignore"` and the module version is too old, `import_optional_dependency` returns `None` instead of returning the module as documented.

## Property-Based Test

```python
from hypothesis import given, strategies as st

from pandas.compat._optional import import_optional_dependency

@given(st.sampled_from(["hypothesis", "pytest", "numpy"]))
def test_import_optional_dependency_ignore_returns_module(module_name):
    result = import_optional_dependency(module_name, errors="ignore", min_version="999.0.0")
    assert result is not None
```

**Failing input**: Any installed module with `errors="ignore"` and unrealistically high `min_version`

## Reproducing the Bug

```python
from pandas.compat._optional import import_optional_dependency

result = import_optional_dependency("hypothesis", errors="ignore", min_version="999.0.0")

print(f"Result: {result}")
print(f"Expected: <module 'hypothesis' ...>")
print(f"Actual: {result}")
```

## Why This Is A Bug

The docstring for `import_optional_dependency` states (lines 110-113):

```
ignore: If the module is not installed, return None, otherwise,
  return the module, even if the version is too old.
  It's expected that users validate the version locally when
  using ``errors="ignore"`` (see. ``io/html.py``)
```

However, the implementation (lines 163-166) returns `None` when the version is too old:

```python
elif errors == "raise":
    raise ImportError(msg)
else:
    return None  # <-- This is when errors="ignore" and version is too old
```

This contradicts the documented behavior.

## Fix

```diff
--- a/pandas/compat/_optional.py
+++ b/pandas/compat/_optional.py
@@ -162,8 +162,6 @@ def import_optional_dependency(
                 return None
             elif errors == "raise":
                 raise ImportError(msg)
-            else:
-                return None

     return module
```