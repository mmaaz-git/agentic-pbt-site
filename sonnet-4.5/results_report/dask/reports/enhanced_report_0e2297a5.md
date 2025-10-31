# Bug Report: dask.base.key_split UnicodeDecodeError on Non-UTF8 Bytes

**Target**: `dask.base.key_split`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `key_split` function crashes with `UnicodeDecodeError` when given bytes that cannot be decoded as UTF-8, despite explicitly supporting bytes inputs in its documentation.

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

# Run the test
test_key_split_returns_string()
```

<details>

<summary>
**Failing input**: `b'\x80'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/35/hypo.py", line 14, in <module>
    test_key_split_returns_string()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/35/hypo.py", line 5, in test_key_split_returns_string
    st.text(min_size=1),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/35/hypo.py", line 10, in test_key_split_returns_string
    result = dask.base.key_split(s)
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/utils.py", line 1979, in key_split
    return key_split(s.decode())
                     ~~~~~~~~^^
UnicodeDecodeError: 'utf-8' codec can't decode byte 0x80 in position 0: invalid start byte
Falsifying example: test_key_split_returns_string(
    s=b'\x80',
)
```
</details>

## Reproducing the Bug

```python
import dask.base

# Test case that crashes with UnicodeDecodeError
result = dask.base.key_split(b'\x80')
print(f"Result: {result}")
```

<details>

<summary>
UnicodeDecodeError when decoding non-UTF8 bytes
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/35/repo.py", line 4, in <module>
    result = dask.base.key_split(b'\x80')
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/utils.py", line 1979, in key_split
    return key_split(s.decode())
                     ~~~~~~~~^^
UnicodeDecodeError: 'utf-8' codec can't decode byte 0x80 in position 0: invalid start byte
```
</details>

## Why This Is A Bug

This violates expected behavior in multiple ways:

1. **Documentation explicitly supports bytes**: The docstring at line 1960 of `/home/npc/miniconda/lib/python3.13/site-packages/dask/utils.py` shows `key_split(b'hello-world-1')` as a valid example, indicating bytes inputs should be supported.

2. **Inconsistent error handling**: The function has robust error handling that returns `'Other'` for various edge cases (lines 2001-2002 catch all exceptions), but the bytes decoding at line 1979 happens before this safety net, causing an unhandled crash.

3. **No UTF-8 requirement documented**: Neither the docstring nor any documentation specifies that bytes must be valid UTF-8. Python bytes objects can contain any byte values (0x00-0xFF), not just UTF-8 encoded text.

4. **Violates principle of graceful degradation**: The function successfully handles `None` (returns `'Other'`), malformed strings, and other edge cases gracefully, but fails catastrophically on non-UTF8 bytes.

## Relevant Context

The `key_split` function is used internally by Dask to extract meaningful prefixes from task keys for grouping and visualization purposes. Task keys can come from various sources including:
- User-provided names
- Generated identifiers
- Serialized data
- File paths or binary data

The function is located in `/home/npc/miniconda/lib/python3.13/site-packages/dask/utils.py` starting at line 1944. The problematic code is at line 1979 where `s.decode()` is called without error handling.

Documentation link: The function is part of Dask's internal utilities for distributed computing task management.

## Proposed Fix

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

This fix uses Python's `errors='replace'` parameter which replaces invalid UTF-8 bytes with the Unicode replacement character (�), allowing the function to continue processing without crashing. This maintains backward compatibility for valid UTF-8 bytes while gracefully handling invalid sequences.