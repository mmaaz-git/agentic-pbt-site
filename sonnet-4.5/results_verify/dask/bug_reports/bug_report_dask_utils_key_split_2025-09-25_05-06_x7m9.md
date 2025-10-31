# Bug Report: dask.utils.key_split Crashes on Invalid UTF-8 Bytes

**Target**: `dask.utils.key_split`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `key_split` function crashes with `UnicodeDecodeError` when given bytes that are not valid UTF-8, despite having a catch-all exception handler that should return "Other" for problematic inputs.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from dask.utils import key_split

@given(st.binary())
def test_key_split_bytes_idempotence(s):
    first_split = key_split(s)
    second_split = key_split(first_split)
    assert first_split == second_split
```

**Failing input**: `b'\x80'`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env')

from dask.utils import key_split

invalid_bytes = b'\x80'
try:
    result = key_split(invalid_bytes)
    print(f"Result: {result}")
except UnicodeDecodeError as e:
    print(f"Crashed with: {e}")
```

## Why This Is A Bug

The function's docstring includes an example with bytes input: `key_split(b'hello-world-1')`, indicating bytes are supported. The function also has a catch-all `except Exception` handler at the end that returns "Other" for problematic inputs, suggesting it's designed to handle edge cases gracefully.

However, the bytes-to-string conversion happens before the exception handler:

```python
if type(s) is bytes:
    return key_split(s.decode())  # Crashes here on invalid UTF-8
```

The `UnicodeDecodeError` occurs before reaching the try-except block, causing the function to crash instead of returning "Other" as intended.

## Fix

Use error handling when decoding bytes:

```diff
--- a/dask/utils.py
+++ b/dask/utils.py
@@ -1977,7 +1977,10 @@ def key_split(s):
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