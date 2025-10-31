# Bug Report: dask.utils.key_split Crashes on Non-UTF8 Bytes

**Target**: `dask.utils.key_split`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `key_split` function crashes with `UnicodeDecodeError` when given byte strings that are not valid UTF-8, despite the function's documentation showing it accepts byte inputs.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from dask.utils import key_split


@given(st.binary())
def test_key_split_bytes(b):
    result = key_split(b)
    assert isinstance(result, str)
```

**Failing input**: `b=b'\x80'`

## Reproducing the Bug

```python
from dask.utils import key_split

invalid_utf8 = b'\x80'
result = key_split(invalid_utf8)
```

Output:
```
UnicodeDecodeError: 'utf-8' codec can't decode byte 0x80 in position 0: invalid start byte
```

## Why This Is A Bug

The function's docstring includes an example showing it accepts byte inputs:
```python
>>> key_split(b'hello-world-1')
'hello-world'
```

This implies the function should handle arbitrary byte strings. However, the implementation assumes all byte inputs are valid UTF-8:

```python
if type(s) is bytes:
    return key_split(s.decode())
```

When the bytes cannot be decoded as UTF-8, the function crashes instead of gracefully handling the error or returning a fallback value like `'Other'` (which it returns for `None` and other exceptional cases).

## Fix

```diff
--- a/dask/utils.py
+++ b/dask/utils.py
@@ -1976,7 +1976,10 @@ def key_split(s):
     """
     # If we convert the key, recurse to utilize LRU cache better
     if type(s) is bytes:
-        return key_split(s.decode())
+        try:
+            return key_split(s.decode())
+        except UnicodeDecodeError:
+            return "Other"
     if type(s) is tuple:
         return key_split(s[0])
     try:
```

This fix makes the function return `'Other'` for non-UTF8 byte strings, consistent with how it handles other unparseable inputs.