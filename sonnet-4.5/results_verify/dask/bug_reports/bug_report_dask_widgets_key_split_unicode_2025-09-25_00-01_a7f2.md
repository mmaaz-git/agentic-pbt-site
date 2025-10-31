# Bug Report: dask.widgets key_split crashes on non-UTF-8 bytes

**Target**: `dask.utils.key_split` (exposed via `dask.widgets.FILTERS['key_split']`)
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `key_split` function crashes with a `UnicodeDecodeError` when given bytes that are not valid UTF-8, despite the function accepting bytes as a documented input type.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from dask.widgets import FILTERS


@given(st.binary(min_size=1))
def test_key_split_bytes_returns_string(b):
    key_split = FILTERS['key_split']
    result = key_split(b)
    assert isinstance(result, str)
```

**Failing input**: `b'\x80'`

## Reproducing the Bug

```python
from dask.utils import key_split

result = key_split(b'\x80')
```

This raises:
```
UnicodeDecodeError: 'utf-8' codec can't decode byte 0x80 in position 0: invalid start byte
```

## Why This Is A Bug

The function's docstring includes an example with bytes input:
```python
>>> key_split(b'hello-world-1')
'hello-world'
```

This suggests that bytes are a valid input type. However, the implementation blindly calls `.decode()` without error handling, assuming all bytes are valid UTF-8. When non-UTF-8 bytes are passed (which are perfectly valid Python bytes objects), the function crashes.

The function should either:
1. Handle non-UTF-8 bytes gracefully (e.g., using `errors='replace'` or `errors='ignore'`)
2. Document that only UTF-8 encoded bytes are supported

## Fix

Use error-tolerant decoding:

```diff
--- a/dask/utils.py
+++ b/dask/utils.py
@@ -1976,7 +1976,7 @@ def key_split(s):
     """
     # If we convert the key, recurse to utilize LRU cache better
     if type(s) is bytes:
-        return key_split(s.decode())
+        return key_split(s.decode('utf-8', errors='replace'))
     if type(s) is tuple:
         return key_split(s[0])
     try:
```

This ensures the function never crashes on invalid UTF-8 bytes, replacing invalid sequences with the Unicode replacement character.