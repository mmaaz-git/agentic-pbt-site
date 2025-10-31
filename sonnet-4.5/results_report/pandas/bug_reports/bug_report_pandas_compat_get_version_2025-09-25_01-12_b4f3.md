# Bug Report: pandas.compat._optional.get_version Returns Non-String for Non-String __version__

**Target**: `pandas.compat._optional.get_version`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`get_version` returns non-string values when a module's `__version__` attribute is not a string, violating its return type annotation and causing downstream crashes.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from pandas.compat._optional import get_version
import types


@given(st.one_of(st.integers(), st.floats(), st.lists(st.integers()), st.none()))
@settings(max_examples=100)
def test_get_version_with_non_string_version(version_value):
    mock_module = types.ModuleType("test_module")
    mock_module.__version__ = version_value

    result = get_version(mock_module)
    assert isinstance(result, str)
```

**Failing input**: `version_value=0` (or any non-string value)

## Reproducing the Bug

```python
from pandas.compat._optional import get_version
import types

mock_module = types.ModuleType("test_module")
mock_module.__version__ = 0

result = get_version(mock_module)
print(f"Result: {result}")
print(f"Type: {type(result)}")
```

Output:
```
Result: 0
Type: <class 'int'>
```

This violates the function's type signature (`-> str`) and will cause crashes when the result is used, e.g., at line 81 where `version.split()` is called for psycopg2.

## Why This Is A Bug

1. The function signature declares `-> str` return type
2. Line 81 calls `version.split()[0]` which requires a string
3. Returning non-string values violates the API contract and causes downstream AttributeError

## Fix

```diff
--- a/pandas/compat/_optional.py
+++ b/pandas/compat/_optional.py
@@ -74,7 +74,7 @@ def get_version(module: types.ModuleType) -> str:
     version = getattr(module, "__version__", None)

     if version is None:
         raise ImportError(f"Can't determine version for {module.__name__}")
+    version = str(version)
     if module.__name__ == "psycopg2":
         # psycopg2 appends " (dt dec pq3 ext lo64)" to it's version
         version = version.split()[0]
```