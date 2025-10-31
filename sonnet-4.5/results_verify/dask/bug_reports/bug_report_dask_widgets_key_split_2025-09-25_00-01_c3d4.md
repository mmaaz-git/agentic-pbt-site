# Bug Report: dask.widgets key_split UnicodeDecodeError on Invalid UTF-8 Bytes

**Target**: `dask.widgets.widgets.key_split`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `key_split` function crashes with `UnicodeDecodeError` when given bytes that are not valid UTF-8, despite the docstring showing it accepts bytes input and the function having a general exception handler that returns `"Other"`.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from dask.widgets.widgets import key_split

@given(st.binary(min_size=0, max_size=100))
def test_key_split_bytes(b):
    result = key_split(b)
    assert isinstance(result, str)
```

**Failing input**: `b = b'\x80'`

## Reproducing the Bug

```python
from dask.widgets.widgets import key_split

print(key_split(b'hello-world-1'))

key_split(b'\x80')
```

## Why This Is A Bug

The function's docstring includes an example showing bytes support:

```python
>>> key_split(b'hello-world-1')
'hello-world'
```

This implies the function should handle bytes input. Additionally, the function has a try-except block that catches `Exception` and returns `"Other"` for any errors, suggesting it should handle edge cases gracefully. However, the bytes-to-string conversion happens *before* the try-except block, causing crashes on invalid UTF-8.

The function also accepts `None` and returns `"Other"`, showing it's designed to be defensive.

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
+            return key_split(s.decode('utf-8', errors='replace'))
+        except Exception:
+            return "Other"
     if type(s) is tuple:
         return key_split(s[0])
     try:
```

Using `errors='replace'` will replace invalid UTF-8 sequences with the Unicode replacement character (U+FFFD), allowing the function to continue processing. Alternatively, `errors='ignore'` could be used to skip invalid bytes entirely.