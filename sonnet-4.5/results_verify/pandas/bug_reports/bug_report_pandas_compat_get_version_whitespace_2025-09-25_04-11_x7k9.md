# Bug Report: pandas.compat._optional.get_version IndexError on Whitespace-Only Version

**Target**: `pandas.compat._optional.get_version`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `get_version` function crashes with an unhelpful `IndexError` when called on a module named "psycopg2" that has a whitespace-only `__version__` string, instead of raising the more appropriate `ImportError` with a clear message.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pandas.compat._optional import get_version
import types

@given(st.text())
def test_get_version_psycopg2_splits_on_whitespace(version_str):
    mock_module = types.ModuleType("psycopg2")
    mock_module.__version__ = version_str
    result = get_version(mock_module)
    assert isinstance(result, str)
    assert result == version_str.split()[0]
```

**Failing input**: `version_str='\r'`

## Reproducing the Bug

```python
from pandas.compat._optional import get_version
import types

mock_module = types.ModuleType("psycopg2")
mock_module.__version__ = "\r"

result = get_version(mock_module)
```

Output:
```
IndexError: list index out of range
```

## Why This Is A Bug

When `get_version` encounters a psycopg2 module, it attempts to split the version string and take the first element:

```python
if module.__name__ == "psycopg2":
    version = version.split()[0]
```

However, if the version string contains only whitespace (e.g., `"\r"`, `"\n"`, `" "`), `split()` returns an empty list, causing `[0]` to raise an `IndexError`.

The function should either:
1. Handle this edge case and raise a clear `ImportError` like it does for missing `__version__`
2. Return the original version string if splitting produces an empty list

## Fix

```diff
--- a/pandas/compat/_optional.py
+++ b/pandas/compat/_optional.py
@@ -78,7 +78,10 @@ def get_version(module: types.ModuleType) -> str:
         raise ImportError(f"Can't determine version for {module.__name__}")
     if module.__name__ == "psycopg2":
         # psycopg2 appends " (dt dec pq3 ext lo64)" to it's version
-        version = version.split()[0]
+        parts = version.split()
+        if not parts:
+            raise ImportError(f"Can't determine version for {module.__name__}")
+        version = parts[0]
     return version
```