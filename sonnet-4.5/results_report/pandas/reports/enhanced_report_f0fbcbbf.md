# Bug Report: pandas.core.computation.common.ensure_decoded UnicodeDecodeError on Invalid UTF-8 Bytes

**Target**: `pandas.core.computation.common.ensure_decoded`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `ensure_decoded` function crashes with `UnicodeDecodeError` when given bytes containing invalid UTF-8 sequences, preventing the reading of HDF5 files with corrupted or non-UTF-8 encoded metadata.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pandas.core.computation.common import ensure_decoded

@given(st.binary())
def test_ensure_decoded_returns_str(data):
    result = ensure_decoded(data)
    assert isinstance(result, str)

if __name__ == "__main__":
    test_ensure_decoded_returns_str()
```

<details>

<summary>
**Failing input**: `b'\x80'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/45/hypo.py", line 10, in <module>
    test_ensure_decoded_returns_str()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/45/hypo.py", line 5, in test_ensure_decoded_returns_str
    def test_ensure_decoded_returns_str(data):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/45/hypo.py", line 6, in test_ensure_decoded_returns_str
    result = ensure_decoded(data)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/computation/common.py", line 15, in ensure_decoded
    s = s.decode(get_option("display.encoding"))
UnicodeDecodeError: 'utf-8' codec can't decode byte 0x80 in position 0: invalid start byte
Falsifying example: test_ensure_decoded_returns_str(
    data=b'\x80',
)
```
</details>

## Reproducing the Bug

```python
from pandas.core.computation.common import ensure_decoded

data = b'\x80'
result = ensure_decoded(data)
print(f"Result: {result}")
```

<details>

<summary>
UnicodeDecodeError when decoding invalid UTF-8 byte sequence
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/45/repo.py", line 4, in <module>
    result = ensure_decoded(data)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/computation/common.py", line 15, in ensure_decoded
    s = s.decode(get_option("display.encoding"))
UnicodeDecodeError: 'utf-8' codec can't decode byte 0x80 in position 0: invalid start byte
```
</details>

## Why This Is A Bug

The `ensure_decoded` function is designed to convert bytes to unicode strings, as stated in its docstring: "If we have bytes, decode them to unicode." However, it fails to handle the case where bytes contain invalid UTF-8 sequences. The function uses strict decoding (the default) which raises `UnicodeDecodeError` on invalid sequences instead of gracefully handling them.

This violates the principle of robustness for I/O functions - they should handle malformed input gracefully rather than crashing. While pandas-generated HDF5 files should contain valid UTF-8, the function may encounter:
- Corrupted HDF5 files due to disk errors or incomplete writes
- HDF5 files created by other tools that use different encodings
- Manually created test files with non-UTF-8 data
- Legacy files from older systems

The function's purpose is to "ensure decoded" - it should fulfill this purpose even with imperfect input, using error handlers like 'replace' or 'ignore' rather than crashing.

## Relevant Context

The `ensure_decoded` function is an internal utility located at `/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/computation/common.py:15`. It's primarily used in the context of reading metadata from HDF5/PyTables files.

Key implementation details:
- The function checks if input is `np.bytes_` or `bytes` type
- It uses `get_option("display.encoding")` which defaults to 'utf-8'
- The decode() method is called without an errors parameter, defaulting to 'strict'
- This causes any invalid byte sequence to raise UnicodeDecodeError

The byte `b'\x80'` is particularly problematic as it's an invalid UTF-8 start byte. In UTF-8:
- Bytes 0x00-0x7F are valid single-byte characters
- Bytes 0x80-0xBF are only valid as continuation bytes
- Starting a sequence with 0x80 is always invalid

## Proposed Fix

```diff
--- a/pandas/core/computation/common.py
+++ b/pandas/core/computation/common.py
@@ -12,7 +12,7 @@ def ensure_decoded(s) -> str:
     If we have bytes, decode them to unicode.
     """
     if isinstance(s, (np.bytes_, bytes)):
-        s = s.decode(get_option("display.encoding"))
+        s = s.decode(get_option("display.encoding"), errors='replace')
     return s
```