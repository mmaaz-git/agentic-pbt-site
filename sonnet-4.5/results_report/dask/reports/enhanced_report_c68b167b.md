# Bug Report: dask.utils.key_split Crashes on Non-UTF-8 Bytes

**Target**: `dask.utils.key_split`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `key_split` function crashes with `UnicodeDecodeError` when processing non-UTF-8 byte sequences, despite explicitly accepting bytes as a documented input type and having error handling that returns "Other" for other problematic inputs.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from dask.utils import key_split


@given(st.binary())
@settings(max_examples=300)
def test_key_split_bytes(b):
    result = key_split(b)
    assert isinstance(result, str), f"key_split should return str for bytes, got {type(result)}"


if __name__ == "__main__":
    test_key_split_bytes()
```

<details>

<summary>
**Failing input**: `b=b'\x80'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/42/hypo.py", line 13, in <module>
    test_key_split_bytes()
    ~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/42/hypo.py", line 6, in test_key_split_bytes
    @settings(max_examples=300)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/42/hypo.py", line 8, in test_key_split_bytes
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

# Test the failing case with non-UTF-8 bytes
result = key_split(b'\x80')
print(f"Result: {result}")
```

<details>

<summary>
UnicodeDecodeError on invalid UTF-8 byte sequence
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/42/repo.py", line 4, in <module>
    result = key_split(b'\x80')
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/utils.py", line 1979, in key_split
    return key_split(s.decode())
                     ~~~~~~~~^^
UnicodeDecodeError: 'utf-8' codec can't decode byte 0x80 in position 0: invalid start byte
```
</details>

## Why This Is A Bug

This violates expected behavior for three key reasons:

1. **Documented bytes support**: The function's docstring explicitly includes a bytes example (`key_split(b'hello-world-1')` → `'hello-world'`), establishing bytes as a supported input type. When a function accepts a type, it should handle all valid instances of that type gracefully.

2. **Inconsistent error handling**: The function has a try-except block (lines 1982-2002) that catches exceptions and returns "Other" for problematic inputs like `None`. However, the bytes decoding on line 1979 occurs *outside* this try block, causing UnicodeDecodeError to escape the intended error handling. This appears to be a code structure oversight rather than intentional design.

3. **Defensive programming pattern violated**: The function demonstrates a clear pattern of defensive programming by returning "Other" for inputs it cannot process. The bytes decoding error breaks this pattern, causing a crash instead of graceful degradation.

## Relevant Context

The `key_split` function is an internal Dask utility decorated with `@functools.lru_cache(100000)`, indicating it's performance-critical and frequently called. It extracts key prefixes from task identifiers in Dask's distributed computing framework.

Key observations from the code (dask/utils.py):
- Lines 1978-1979: Bytes handling is explicit but unprotected
- Lines 1982-2002: General exception handler exists but doesn't catch decode errors
- Line 1970-1971: Docstring shows `key_split(None)` returns "Other"
- Line 1964-1965: Docstring shows bytes are supported inputs

In distributed computing contexts, task keys might come from various sources and could contain arbitrary byte sequences, making robust handling important for system stability.

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