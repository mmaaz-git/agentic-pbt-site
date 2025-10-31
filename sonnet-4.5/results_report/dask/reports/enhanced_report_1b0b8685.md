# Bug Report: dask.utils.key_split crashes on non-UTF-8 bytes

**Target**: `dask.utils.key_split`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `key_split` function crashes with a `UnicodeDecodeError` when given bytes that are not valid UTF-8, despite its docstring explicitly demonstrating support for bytes input without any encoding restrictions.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from dask.utils import key_split

@given(st.binary())
def test_key_split_bytes(b):
    result = key_split(b)
    assert isinstance(result, str)

if __name__ == "__main__":
    test_key_split_bytes()
```

<details>

<summary>
**Failing input**: `b'\x80'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/45/hypo.py", line 10, in <module>
    test_key_split_bytes()
    ~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/45/hypo.py", line 5, in test_key_split_bytes
    def test_key_split_bytes(b):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/45/hypo.py", line 6, in test_key_split_bytes
    result = key_split(b)
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/utils.py", line 1979, in key_split
    return key_split(s.decode())
                     ~~~~~~~~^^
UnicodeDecodeError: 'utf-8' codec can't decode byte 0x80 in position 0: invalid start byte
Falsifying example: test_key_split_bytes(
    b=b'\x80',
)
```
</details>

## Reproducing the Bug

```python
from dask.utils import key_split

invalid_utf8_bytes = b'\x80'
result = key_split(invalid_utf8_bytes)
print(f"Result: {result}")
```

<details>

<summary>
UnicodeDecodeError: 'utf-8' codec can't decode byte 0x80 in position 0
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/45/repo.py", line 4, in <module>
    result = key_split(invalid_utf8_bytes)
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/utils.py", line 1979, in key_split
    return key_split(s.decode())
                     ~~~~~~~~^^
UnicodeDecodeError: 'utf-8' codec can't decode byte 0x80 in position 0: invalid start byte
```
</details>

## Why This Is A Bug

The function's docstring explicitly includes an example demonstrating bytes input: `>>> key_split(b'hello-world-1')` returns `'hello-world'`. This creates a clear contract that bytes are a supported input type. However, the implementation has a critical flaw at line 1979 where it unconditionally calls `s.decode()` without any error handling or encoding specification.

The bug violates the documented behavior in several ways:
1. **No encoding restrictions documented**: The docstring shows bytes support without mentioning UTF-8 requirements
2. **Inconsistent error handling**: The function has a try/except block for other errors (line 1982-2002) but fails to handle decode errors
3. **Incomplete type support**: While the function checks for bytes type (line 1978), it doesn't handle the full range of valid bytes values (0x00-0xFF)

Since Dask is used in distributed computing scenarios where task keys might include serialized binary data or non-UTF-8 encoded strings, this crash represents a real failure case that users could encounter.

## Relevant Context

The `key_split` function is decorated with `@functools.lru_cache(100000)` (line 1947), suggesting it's frequently called and performance-critical. It's designed to extract human-readable names from Dask task keys for debugging and visualization purposes.

The function already handles various edge cases:
- Returns "Other" for exceptions during string processing (line 2002)
- Handles None input gracefully (returns "Other")
- Processes tuples by recursing on the first element
- Strips hex patterns and special characters from keys

Code location: `/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages/dask/utils.py:1948-2002`

## Proposed Fix

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
         return key_split(s[0])
     try:
```