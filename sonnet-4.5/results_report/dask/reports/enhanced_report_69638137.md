# Bug Report: dask.utils.key_split Crashes on Invalid UTF-8 Bytes

**Target**: `dask.utils.key_split`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `key_split` function crashes with `UnicodeDecodeError` when given bytes that are not valid UTF-8, despite having explicit support for bytes input and a catch-all exception handler designed to return "Other" for problematic inputs.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
from dask.utils import key_split

@given(st.binary())
def test_key_split_bytes_idempotence(s):
    first_split = key_split(s)
    second_split = key_split(first_split)
    assert first_split == second_split

if __name__ == "__main__":
    test_key_split_bytes_idempotence()
```

<details>

<summary>
**Failing input**: `b'\x80'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/12/hypo.py", line 14, in <module>
    test_key_split_bytes_idempotence()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/12/hypo.py", line 8, in test_key_split_bytes_idempotence
    def test_key_split_bytes_idempotence(s):
                   ^^^
  File "/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/12/hypo.py", line 9, in test_key_split_bytes_idempotence
    first_split = key_split(s)
  File "/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages/dask/utils.py", line 1979, in key_split
    return key_split(s.decode())
                     ~~~~~~~~^^
UnicodeDecodeError: 'utf-8' codec can't decode byte 0x80 in position 0: invalid start byte
Falsifying example: test_key_split_bytes_idempotence(
    s=b'\x80',
)
```
</details>

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages')

from dask.utils import key_split

# Test with invalid UTF-8 bytes
invalid_bytes = b'\x80'
try:
    result = key_split(invalid_bytes)
    print(f"Result: {result}")
except UnicodeDecodeError as e:
    print(f"Crashed with UnicodeDecodeError: {e}")
except Exception as e:
    print(f"Crashed with other exception: {type(e).__name__}: {e}")
```

<details>

<summary>
UnicodeDecodeError crash
</summary>
```
Crashed with UnicodeDecodeError: 'utf-8' codec can't decode byte 0x80 in position 0: invalid start byte
```
</details>

## Why This Is A Bug

This violates expected behavior in three critical ways:

1. **Documented bytes support**: The function's docstring at line 1964 explicitly includes an example with bytes input: `>>> key_split(b'hello-world-1')`, clearly indicating that bytes are a supported input type. The documentation makes no mention that these bytes must be valid UTF-8.

2. **Broken exception handling contract**: The function has a comprehensive catch-all exception handler (lines 2001-2002) that returns "Other" for any problematic input. The docstring even demonstrates this with `>>> key_split(None)` returning `'Other'`. This strongly indicates the function is designed to handle edge cases gracefully without crashing.

3. **Structural bug in error handling**: The bytes-to-string conversion at line 1979 (`return key_split(s.decode())`) occurs BEFORE the try-except block that starts at line 1982. This means the `UnicodeDecodeError` is raised outside the exception handler's scope, defeating the function's defensive programming design.

The bug breaks the idempotence property that `key_split(key_split(x))` should equal `key_split(x)` for all valid inputs, as the function can crash on certain byte sequences that could theoretically be returned by the function itself (though "Other" is typically returned for edge cases).

## Relevant Context

The `key_split` function is decorated with `@functools.lru_cache(100000)` for performance optimization, which makes the recursive call pattern for type conversion efficient. The function is designed to extract meaningful prefixes from Dask task keys, handling various input formats including strings, tuples, bytes, and edge cases.

The function's location in the codebase: `/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages/dask/utils.py`

Invalid UTF-8 byte sequences that trigger this bug include:
- Continuation bytes without proper start bytes (0x80-0xBF)
- Invalid start bytes (0xC0-0xC1, 0xF5-0xFF)
- Incomplete multi-byte sequences
- Overlong encodings
- Surrogate pair bytes

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
+            return key_split(s.decode())
+        except UnicodeDecodeError:
+            return "Other"
     if type(s) is tuple:
         return key_split(s[0])
     try:
```