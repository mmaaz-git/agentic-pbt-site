# Bug Report: xarray.backends.netcdf3.is_valid_nc3_name Empty String Crash

**Target**: `xarray.backends.netcdf3.is_valid_nc3_name`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`is_valid_nc3_name()` crashes with `IndexError` when passed an empty string, instead of returning `False`.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from xarray.backends.netcdf3 import is_valid_nc3_name

@given(st.text())
def test_is_valid_nc3_name_doesnt_crash(s):
    result = is_valid_nc3_name(s)
    assert isinstance(result, bool)
```

**Failing input**: `s=''`

## Reproducing the Bug

```python
from xarray.backends.netcdf3 import is_valid_nc3_name

result = is_valid_nc3_name("")
```

Output:
```
IndexError: string index out of range
```

## Why This Is A Bug

The function is documented to "test whether an object can be validly converted to a netCDF-3 dimension, variable or attribute name". An empty string should return `False`, not crash. The function tries to access `s[-1]` (to check for trailing spaces) without first verifying that the string is non-empty.

## Fix

```diff
--- a/xarray/backends/netcdf3.py
+++ b/xarray/backends/netcdf3.py
@@ -159,7 +159,8 @@ def is_valid_nc3_name(s):
     if not isinstance(s, str):
         return False
     num_bytes = len(s.encode("utf-8"))
     return (
+        (len(s) > 0)
+        and (unicodedata.normalize("NFC", s) == s)
-        (unicodedata.normalize("NFC", s) == s)
         and (s not in _reserved_names)
         and (num_bytes >= 0)
         and ("/" not in s)
```

Alternatively, the check `(s[-1] != " ")` could be replaced with `(not s.endswith(" "))` which works correctly for empty strings.