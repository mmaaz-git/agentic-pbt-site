# Bug Report: dask.utils.key_split UnicodeDecodeError

**Target**: `dask.utils.key_split`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `key_split` function crashes with a `UnicodeDecodeError` when passed bytes that are not valid UTF-8, despite the docstring showing bytes as a valid input type and the function having general exception handling.

## Property-Based Test

```python
from hypothesis import given
import hypothesis.strategies as st
from dask.utils import key_split


@given(st.binary(min_size=1, max_size=50))
def test_key_split_with_bytes(b):
    result = key_split(b)
    assert isinstance(result, str)
```

**Failing input**: `b'\x80'`

## Reproducing the Bug

```python
from dask.utils import key_split

print(key_split(b'hello-world-1'))

print(key_split(b'\x80'))
```

**Output:**
```
'hello-world'
Traceback (most recent call last):
  ...
UnicodeDecodeError: 'utf-8' codec can't decode byte 0x80 in position 0: invalid start byte
```

## Why This Is A Bug

1. The function's docstring explicitly shows bytes as a valid input:
   ```python
   >>> key_split(b'hello-world-1')
   'hello-world'
   ```

2. The function has a general exception handler at the end that returns `"Other"` for unexpected inputs, suggesting robustness is intended:
   ```python
   try:
       # ... processing logic
   except Exception:
       return "Other"
   ```

3. The `UnicodeDecodeError` occurs in the bytes handling code before the exception handler, making the function inconsistent in its error handling.

4. The function is used in critical paths (visualization, optimization) where crashes could affect user workflows.

## Fix

The fix is to handle `UnicodeDecodeError` when decoding bytes. Change line 1979 in `/dask/utils.py`:

```diff
     if type(s) is bytes:
-        return key_split(s.decode())
+        try:
+            return key_split(s.decode())
+        except UnicodeDecodeError:
+            return key_split(s.decode(errors='replace'))
```

Alternatively, use `errors='replace'` or `errors='ignore'` directly:

```diff
     if type(s) is bytes:
-        return key_split(s.decode())
+        return key_split(s.decode(errors='replace'))
```