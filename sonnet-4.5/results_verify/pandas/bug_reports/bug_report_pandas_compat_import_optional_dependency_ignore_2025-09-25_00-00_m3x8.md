# Bug Report: pandas.compat._optional.import_optional_dependency errors='ignore' Returns None for Old Versions

**Target**: `pandas.compat._optional.import_optional_dependency`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

When `errors='ignore'` and a module's version is too old, `import_optional_dependency()` returns `None` instead of returning the module, violating its documented contract.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st
from pandas.compat._optional import import_optional_dependency

@given(min_version=st.text(min_size=1).filter(lambda s: s[0].isdigit()))
@settings(max_examples=100)
def test_errors_ignore_returns_module_even_if_old(min_version):
    result = import_optional_dependency("numpy", min_version="999.0.0", errors="ignore")
    assert result is not None, (
        "errors='ignore' should return module even if version is too old"
    )
```

**Failing input**: `min_version="999.0.0"`

## Reproducing the Bug

```python
from pandas.compat._optional import import_optional_dependency

result = import_optional_dependency("numpy", min_version="999.0.0", errors="ignore")
print(f"Result: {result}")
```

**Output**:
```
Result: None
```

The function returns `None` even though numpy is installed.

## Why This Is A Bug

The docstring explicitly states (lines 110-113):

```python
* ignore: If the module is not installed, return None, otherwise,
  return the module, even if the version is too old.
  It's expected that users validate the version locally when
  using ``errors="ignore"`` (see. ``io/html.py``)
```

The phrase "return the module, even if the version is too old" clearly indicates that when `errors='ignore'`, the function should return the imported module regardless of version.

However, the implementation (lines 163-166) returns `None`:

```python
elif errors == "raise":
    raise ImportError(msg)
else:
    return None  # This is the errors='ignore' case
```

This violates the contract. When `errors='ignore'`, users expect to get the module and do their own version validation, but they get `None` instead.

## Fix

```diff
--- a/pandas/compat/_optional.py
+++ b/pandas/compat/_optional.py
@@ -163,7 +163,7 @@ def import_optional_dependency(
             elif errors == "raise":
                 raise ImportError(msg)
             else:
-                return None
+                pass  # errors='ignore', continue and return module below

     return module
```

With this fix, when `errors='ignore'` and the version is too old, the function will skip the version error handling and proceed to return the module at line 168.