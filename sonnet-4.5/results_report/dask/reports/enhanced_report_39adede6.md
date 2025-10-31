# Bug Report: dask.utils.key_split crashes on non-UTF-8 bytes

**Target**: `dask.utils.key_split`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `key_split` function crashes with a `UnicodeDecodeError` when given bytes that are not valid UTF-8, despite the function's docstring explicitly showing bytes as a supported input type.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from dask.widgets import FILTERS


@given(st.binary(min_size=1))
def test_key_split_bytes_returns_string(b):
    key_split = FILTERS['key_split']
    result = key_split(b)
    assert isinstance(result, str)

# Run the test
if __name__ == "__main__":
    test_key_split_bytes_returns_string()
```

<details>

<summary>
**Failing input**: `b'\x80'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/48/hypo.py", line 13, in <module>
    test_key_split_bytes_returns_string()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/48/hypo.py", line 6, in test_key_split_bytes_returns_string
    def test_key_split_bytes_returns_string(b):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/48/hypo.py", line 8, in test_key_split_bytes_returns_string
    result = key_split(b)
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/utils.py", line 1979, in key_split
    return key_split(s.decode())
                     ~~~~~~~~^^
UnicodeDecodeError: 'utf-8' codec can't decode byte 0x80 in position 0: invalid start byte
Falsifying example: test_key_split_bytes_returns_string(
    b=b'\x80',
)
```
</details>

## Reproducing the Bug

```python
from dask.utils import key_split

# Test with non-UTF-8 bytes
try:
    result = key_split(b'\x80')
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

# Test with valid UTF-8 bytes for comparison
try:
    result = key_split(b'hello-world-1')
    print(f"Valid UTF-8 result: {result}")
except Exception as e:
    print(f"Error with valid UTF-8: {type(e).__name__}: {e}")
```

<details>

<summary>
UnicodeDecodeError when processing non-UTF-8 bytes
</summary>
```
Error: UnicodeDecodeError: 'utf-8' codec can't decode byte 0x80 in position 0: invalid start byte
Valid UTF-8 result: hello-world
```
</details>

## Why This Is A Bug

This violates expected behavior because the function's docstring at line 1964 in `/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages/dask/utils.py` explicitly demonstrates bytes as a supported input type with the example:

```python
>>> key_split(b'hello-world-1')
'hello-world'
```

The implementation at line 1978-1979 has explicit handling for bytes type:
```python
if type(s) is bytes:
    return key_split(s.decode())
```

However, the `.decode()` call assumes all bytes are valid UTF-8. When non-UTF-8 bytes are passed (which are perfectly valid Python bytes objects), the function crashes with an unhandled `UnicodeDecodeError`. This contradicts the documented behavior that bytes are a first-class supported input type. The function should either handle decoding errors gracefully or document that only UTF-8 encoded bytes are supported.

## Relevant Context

The `key_split` function is used internally by Dask for visualization and widget display purposes. It's exposed publicly through `dask.widgets.FILTERS['key_split']` (defined in `/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages/dask/widgets/widgets.py` line 18) and is decorated with `@functools.lru_cache(100000)` for performance.

The function's purpose is to extract meaningful prefixes from Dask task keys for display. While most task keys in practice will be strings or UTF-8 encoded bytes, the function accepts bytes as a documented input type and should handle all valid bytes objects without crashing.

The crash location is at line 1979 in `dask/utils.py` where `s.decode()` is called without error handling.

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