# Bug Report: dask.utils.key_split crashes on non-UTF-8 bytes

**Target**: `dask.utils.key_split`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `key_split` function crashes with a `UnicodeDecodeError` when given bytes that are not valid UTF-8, despite documenting support for bytes input in its docstring examples.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from dask.utils import key_split

@given(st.binary())
def test_key_split_bytes(b):
    result = key_split(b)
    assert isinstance(result, str)
```

**Failing input**: `b'\x80'`

## Reproducing the Bug

```python
from dask.utils import key_split

invalid_utf8_bytes = b'\x80'
result = key_split(invalid_utf8_bytes)
```

Output:
```
UnicodeDecodeError: 'utf-8' codec can't decode byte 0x80 in position 0: invalid start byte
```

## Why This Is A Bug

The function's docstring includes an example showing it accepts bytes input:
```python
>>> key_split(b'hello-world-1')
'hello-world'
```

This sets the expectation that arbitrary bytes are supported. However, the implementation blindly calls `.decode()` without error handling, causing crashes on non-UTF-8 byte sequences. Since task keys in dask could theoretically include binary data, this is a realistic failure case.

## Fix

Handle decode errors gracefully by either using error-tolerant decoding or catching the exception:

```diff
--- a/dask/utils.py
+++ b/dask/utils.py
@@ -1976,7 +1976,10 @@ def key_split(s):
     """
     # If we convert the key, recurse to utilize LRU cache better
     if type(s) is bytes:
-        return key_split(s.decode())
+        try:
+            return key_split(s.decode('utf-8', errors='replace'))
+        except Exception:
+            return "Other"
     if type(s) is tuple:
         return key_split(s[0])
     try:
```

This uses `errors='replace'` to substitute invalid UTF-8 sequences with replacement characters, allowing the function to continue processing.