# Bug Report: pandas.compat._optional.import_optional_dependency ValueError on Invalid Module Names

**Target**: `pandas.compat._optional.import_optional_dependency`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`import_optional_dependency` with `errors="ignore"` raises `ValueError` or `TypeError` for invalid module names instead of returning `None` as documented.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from pandas.compat._optional import import_optional_dependency
import types


@given(st.text())
@settings(max_examples=100)
def test_import_optional_dependency_with_existing_module(module_name):
    import sys
    if module_name in sys.modules or '.' in module_name:
        return

    result_ignore = import_optional_dependency(module_name, errors="ignore")
    result_warn = import_optional_dependency(module_name, errors="warn")

    assert result_ignore is None or isinstance(result_ignore, types.ModuleType)
    assert result_warn is None or isinstance(result_warn, types.ModuleType)
```

**Failing input**: `module_name=''` (empty string)

## Reproducing the Bug

```python
from pandas.compat._optional import import_optional_dependency

result = import_optional_dependency("", errors="ignore")
```

This raises:
```
ValueError: Empty module name
```

Similarly:
```python
result = import_optional_dependency("...", errors="ignore")
```

This raises:
```
TypeError: the 'package' argument is required to perform a relative import for '...'
```

## Why This Is A Bug

The function's docstring states that with `errors='ignore'`:
> "If the module is not installed, return None, otherwise, return the module"

Invalid module names (empty strings, relative imports without package) are clearly "not installed" modules, so according to the API contract, they should return `None` rather than raising exceptions.

The function only catches `ImportError` (line 136), but `importlib.import_module` can raise other exceptions like `ValueError` and `TypeError` for malformed module names.

## Fix

```diff
--- a/pandas/compat/_optional.py
+++ b/pandas/compat/_optional.py
@@ -133,7 +133,7 @@ def import_optional_dependency(
     )
     try:
         module = importlib.import_module(name)
-    except ImportError:
+    except (ImportError, ValueError, TypeError):
         if errors == "raise":
             raise ImportError(msg)
         return None
```