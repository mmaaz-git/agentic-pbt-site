# Bug Report: dask.utils.key_split UnicodeDecodeError on Invalid UTF-8 Bytes

**Target**: `dask.utils.key_split`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `key_split` function crashes with a `UnicodeDecodeError` when passed bytes that are not valid UTF-8, despite the docstring explicitly showing bytes as a valid input type and the function having exception handling designed to catch all errors.

## Property-Based Test

```python
from hypothesis import given
import hypothesis.strategies as st
from dask.utils import key_split


@given(st.binary(min_size=1, max_size=50))
def test_key_split_with_bytes(b):
    result = key_split(b)
    assert isinstance(result, str)


if __name__ == "__main__":
    test_key_split_with_bytes()
```

<details>

<summary>
**Failing input**: `b'\x80'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/16/hypo.py", line 13, in <module>
    test_key_split_with_bytes()
    ~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/16/hypo.py", line 7, in test_key_split_with_bytes
    def test_key_split_with_bytes(b):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/16/hypo.py", line 8, in test_key_split_with_bytes
    result = key_split(b)
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/utils.py", line 1979, in key_split
    return key_split(s.decode())
                     ~~~~~~~~^^
UnicodeDecodeError: 'utf-8' codec can't decode byte 0x80 in position 0: invalid start byte
Falsifying example: test_key_split_with_bytes(
    b=b'\x80',
)
```
</details>

## Reproducing the Bug

```python
from dask.utils import key_split

# First test with valid UTF-8 bytes (should work)
print("Testing with valid UTF-8 bytes: b'hello-world-1'")
result = key_split(b'hello-world-1')
print(f"Result: {result}")
print()

# Now test with invalid UTF-8 bytes (will crash)
print("Testing with invalid UTF-8 bytes: b'\\x80'")
result = key_split(b'\x80')
print(f"Result: {result}")
```

<details>

<summary>
UnicodeDecodeError when processing invalid UTF-8 bytes
</summary>
```
Testing with valid UTF-8 bytes: b'hello-world-1'
Result: hello-world

Testing with invalid UTF-8 bytes: b'\x80'
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/16/repo.py", line 11, in <module>
    result = key_split(b'\x80')
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/utils.py", line 1979, in key_split
    return key_split(s.decode())
                     ~~~~~~~~^^
UnicodeDecodeError: 'utf-8' codec can't decode byte 0x80 in position 0: invalid start byte
```
</details>

## Why This Is A Bug

This violates expected behavior for several critical reasons:

1. **Documented Support for Bytes**: The function's docstring explicitly demonstrates bytes as a valid input type with the example `key_split(b'hello-world-1')` returning `'hello-world'`. There is no caveat that only UTF-8 valid bytes are supported.

2. **Inconsistent Exception Handling**: The function contains a general exception handler (lines 1982-2002) that catches all exceptions and returns `"Other"` for unexpected inputs. However, the bytes handling code (lines 1978-1979) is placed OUTSIDE this try-except block, making it inconsistent with the function's design philosophy of graceful error handling.

3. **Decorator Cache Pollution**: The function uses `@functools.lru_cache(100000)` for performance. When it crashes, it doesn't cache the failure, but it also doesn't handle the error gracefully as intended by the exception handler.

4. **Critical Path Usage**: The `key_split` function is used throughout Dask's codebase for task visualization, optimization, and string representations. A crash here can affect multiple user-facing features including the web GUI task display and task fusion operations.

5. **Python Bytes Contract**: Python bytes objects do not guarantee UTF-8 encoding. Requiring UTF-8 validity without documentation violates the principle of least surprise, especially when the function already has mechanisms to handle problematic inputs.

## Relevant Context

The issue stems from a structural problem in the code at `/home/npc/miniconda/lib/python3.13/site-packages/dask/utils.py`:

- Lines 1978-1979: Bytes handling that calls `s.decode()` without error handling
- Lines 1982-2002: The try-except block that is meant to catch all exceptions
- Line 1947: The LRU cache decorator that expects consistent behavior

The function is part of Dask's core utilities and is used by:
- Task visualization in the distributed scheduler dashboard
- String representations of Dask collections (`dask.array`, `dask.dataframe`, etc.)
- Task graph optimization and fusion algorithms
- Task naming and grouping operations

Documentation: https://docs.dask.org/en/stable/api.html#dask.utils.key_split

## Proposed Fix

Move the bytes handling inside the try-except block or add explicit error handling for UnicodeDecodeError:

```diff
--- a/dask/utils.py
+++ b/dask/utils.py
@@ -1975,10 +1975,11 @@ def key_split(s):
     'x'
     """
     # If we convert the key, recurse to utilize LRU cache better
-    if type(s) is bytes:
-        return key_split(s.decode())
-    if type(s) is tuple:
-        return key_split(s[0])
     try:
+        if type(s) is bytes:
+            return key_split(s.decode('utf-8', errors='replace'))
+        if type(s) is tuple:
+            return key_split(s[0])
         words = s.split("-")
         if not words[0][0].isalpha():
             result = words[0].split(",")[0].strip("_'()\"")
```