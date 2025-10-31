# Bug Report: dask.base.key_split UnicodeDecodeError

**Target**: `dask.base.key_split`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`key_split` crashes with `UnicodeDecodeError` when given bytes that are not valid UTF-8, despite explicitly supporting bytes inputs in its docstring.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import dask.base

@given(st.one_of(
    st.text(min_size=1),
    st.binary(min_size=1),
    st.tuples(st.text(min_size=1), st.integers()),
))
def test_key_split_returns_string(s):
    result = dask.base.key_split(s)
    assert isinstance(result, str)
```

**Failing input**: `b'\x80'`

## Reproducing the Bug

```python
import dask.base

dask.base.key_split(b'\x80')
```

Output:
```
UnicodeDecodeError: 'utf-8' codec can't decode byte 0x80 in position 0: invalid start byte
```

## Why This Is A Bug

1. The docstring explicitly shows bytes as supported: `key_split(b'hello-world-1')` → `'hello-world'`
2. The function handles edge cases gracefully (e.g., `None` → `'Other'`)
3. However, it crashes on non-UTF-8 bytes instead of handling them robustly
4. This is inconsistent with the function's design philosophy of handling various input types

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
+            return key_split(s.decode('utf-8'))
+        except UnicodeDecodeError:
+            return key_split(s.decode('utf-8', errors='replace'))
     if type(s) is tuple:
         s = s[0]
     if type(s) is str:
```

Alternative fix: use `errors='replace'` or `errors='ignore'` directly:
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
         s = s[0]
     if type(s) is str:
```