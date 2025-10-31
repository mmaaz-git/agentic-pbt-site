# Bug Report: dask.utils.key_split Crashes on Non-UTF8 Byte Sequences

**Target**: `dask.utils.key_split`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `key_split` function crashes with `UnicodeDecodeError` when processing byte strings containing invalid UTF-8 sequences, despite the function's docstring explicitly demonstrating bytes as a valid input type.

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
**Failing input**: `b=b'\x80'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/7/hypo.py", line 12, in <module>
    test_key_split_bytes()
    ~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/7/hypo.py", line 6, in test_key_split_bytes
    def test_key_split_bytes(b):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/7/hypo.py", line 7, in test_key_split_bytes
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

# Test with invalid UTF-8 byte sequence
invalid_utf8 = b'\x80'
print(f"Testing key_split with invalid UTF-8 bytes: {invalid_utf8!r}")
try:
    result = key_split(invalid_utf8)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print("\n" + "="*50 + "\n")

# Additional test with another invalid UTF-8 sequence
invalid_utf8_2 = b'\xc3\x28'
print(f"Testing key_split with another invalid UTF-8 sequence: {invalid_utf8_2!r}")
try:
    result = key_split(invalid_utf8_2)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print("\n" + "="*50 + "\n")

# Test with valid UTF-8 bytes for comparison
valid_utf8 = b'hello-world-1'
print(f"Testing key_split with valid UTF-8 bytes: {valid_utf8!r}")
try:
    result = key_split(valid_utf8)
    print(f"Result: {result!r}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
```

<details>

<summary>
UnicodeDecodeError crashes on invalid UTF-8 bytes
</summary>
```
Testing key_split with invalid UTF-8 bytes: b'\x80'
Error: UnicodeDecodeError: 'utf-8' codec can't decode byte 0x80 in position 0: invalid start byte

==================================================

Testing key_split with another invalid UTF-8 sequence: b'\xc3('
Error: UnicodeDecodeError: 'utf-8' codec can't decode byte 0xc3 in position 0: invalid continuation byte

==================================================

Testing key_split with valid UTF-8 bytes: b'hello-world-1'
Result: 'hello-world'
```
</details>

## Why This Is A Bug

The function violates its implicit contract in three critical ways:

1. **Documentation explicitly shows bytes as valid input**: The docstring at line 1964 demonstrates `>>> key_split(b'hello-world-1')` returning `'hello-world'`, establishing that bytes are an expected input type. There is no documentation stating that only UTF-8 valid bytes are supported.

2. **Inconsistent error handling**: The function has a general exception handler (lines 2001-2002) that catches all exceptions and returns `"Other"` for unparseable inputs. However, the byte decoding at line 1979 (`return key_split(s.decode())`) happens **outside** the try block, creating an inconsistent error handling pattern where some errors crash while others return `"Other"`.

3. **Unexpected crash in production scenarios**: Python bytes objects can contain any byte values (0x00-0xFF), not just valid UTF-8. Binary data, encrypted content, or data from external sources may contain non-UTF-8 sequences. The function already handles `None` input gracefully by returning `"Other"`, so crashing on certain byte inputs is inconsistent.

## Relevant Context

The `key_split` function is an internal Dask utility (not in the public API documentation) decorated with `@functools.lru_cache(100000)` for performance. Its purpose is to extract meaningful identifiers from various input formats including strings, tuples, and bytes.

The function's existing behavior for edge cases:
- `key_split(None)` returns `"Other"`
- Any exception during string parsing returns `"Other"`
- Valid UTF-8 bytes like `b'hello-world-1'` work correctly

The bug affects any code path in Dask that might pass arbitrary byte sequences to this function, potentially causing unexpected crashes in data processing pipelines.

Source code location: `/lib/python3.13/site-packages/dask/utils.py` lines 1947-2002

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