# Bug Report: pandas.compat._optional.get_version Whitespace Handling

**Target**: `pandas.compat._optional.get_version`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `get_version()` function incorrectly handles psycopg2 version strings that contain whitespace characters other than spaces (e.g., `\r`, `\n`, `\t`) in the version portion, splitting on any whitespace instead of just the first space.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pandas.compat._optional import get_version
import types

@given(
    version=st.text(min_size=1, max_size=20).filter(lambda x: ' ' not in x)
)
def test_get_version_psycopg2_special_case(version):
    mock_module = types.ModuleType("psycopg2")
    mock_module.__version__ = f"{version} (dt dec pq3 ext lo64)"

    result = get_version(mock_module)
    assert result == version
```

**Failing input**: `version='\r'`

## Reproducing the Bug

```python
import types
from pandas.compat._optional import get_version

mock_module = types.ModuleType("psycopg2")
mock_module.__version__ = "\r (dt dec pq3 ext lo64)"

result = get_version(mock_module)
print(f"Expected: {repr('\\r')}")
print(f"Actual: {repr(result)}")
```

Expected output: `'\r'`
Actual output: `'(dt'`

## Why This Is A Bug

The function uses `.split()` which splits on ANY whitespace character, not just spaces. When the version portion contains whitespace characters like `\r`, `\n`, or `\t`, the function incorrectly splits at those characters instead of at the space before the parenthetical metadata. The code comment states that psycopg2 appends " (dt dec pq3 ext lo64)" (note the space), implying the split should happen at that specific space separator, not at any whitespace.

## Fix

```diff
--- a/pandas/compat/_optional.py
+++ b/pandas/compat/_optional.py
@@ -78,7 +78,7 @@ def get_version(module: types.ModuleType) -> str:
         raise ImportError(f"Can't determine version for {module.__name__}")
     if module.__name__ == "psycopg2":
         # psycopg2 appends " (dt dec pq3 ext lo64)" to it's version
-        version = version.split()[0]
+        version = version.split(' ', 1)[0]
     return version
```