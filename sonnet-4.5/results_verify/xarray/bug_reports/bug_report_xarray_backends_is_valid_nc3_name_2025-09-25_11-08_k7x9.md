# Bug Report: xarray.backends.netcdf3.is_valid_nc3_name Empty String Crash

**Target**: `xarray.backends.netcdf3.is_valid_nc3_name`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `is_valid_nc3_name` function crashes with an `IndexError` when given an empty string, instead of returning `False` as expected for an invalid name.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from xarray.backends.netcdf3 import is_valid_nc3_name

@given(st.text())
@settings(max_examples=1000)
def test_is_valid_nc3_name_does_not_crash(name):
    result = is_valid_nc3_name(name)
    assert isinstance(result, bool)
```

**Failing input**: `''` (empty string)

## Reproducing the Bug

```python
from xarray.backends.netcdf3 import is_valid_nc3_name

result = is_valid_nc3_name("")
```

**Output:**
```
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "xarray/backends/netcdf3.py", line 167, in is_valid_nc3_name
    and (s[-1] != " ")
         ^^^^^
IndexError: string index out of range
```

## Why This Is A Bug

The function attempts to check if the last character is a space (`s[-1] != " "`) without first verifying that the string is non-empty. When `s` is an empty string, accessing `s[-1]` raises an `IndexError`.

The bug is on line 167 of `netcdf3.py`:

```python
def is_valid_nc3_name(s):
    if not isinstance(s, str):
        return False
    num_bytes = len(s.encode("utf-8"))
    return (
        (unicodedata.normalize("NFC", s) == s)
        and (s not in _reserved_names)
        and (num_bytes >= 0)
        and ("/" not in s)
        and (s[-1] != " ")       # ← Crashes on empty string
        and (_isalnumMUTF8(s[0]) or (s[0] == "_"))
        and all(_isalnumMUTF8(c) or c in _specialchars for c in s)
    )
```

Empty strings should not be valid netCDF-3 names, so the function should return `False` rather than crash. This is a defensive programming issue - the function should handle all string inputs gracefully.

## Fix

```diff
--- a/xarray/backends/netcdf3.py
+++ b/xarray/backends/netcdf3.py
@@ -159,11 +159,12 @@ def is_valid_nc3_name(s):
     if not isinstance(s, str):
         return False
     num_bytes = len(s.encode("utf-8"))
     return (
+        (num_bytes > 0)
+        and (unicodedata.normalize("NFC", s) == s)
-        (unicodedata.normalize("NFC", s) == s)
         and (s not in _reserved_names)
-        and (num_bytes >= 0)
         and ("/" not in s)
         and (s[-1] != " ")
         and (_isalnumMUTF8(s[0]) or (s[0] == "_"))
         and all(_isalnumMUTF8(c) or c in _specialchars for c in s)
     )
```

This fix:
1. Changes `num_bytes >= 0` to `num_bytes > 0` to reject empty strings
2. Moves this check to the beginning of the boolean expression, so it short-circuits before attempting to access `s[-1]` or `s[0]`
3. Also removes the redundant `num_bytes >= 0` check (len() is always >= 0)