# Bug Report: pandas.core.dtypes.common.ensure_str Invalid UTF-8 Decode Crash

**Target**: `pandas.core.dtypes.common.ensure_str`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `ensure_str` function crashes with a `UnicodeDecodeError` when given bytes containing invalid UTF-8 sequences, violating its documented contract to convert any bytes object to a string.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st
from pandas.core.dtypes.common import ensure_str

@settings(max_examples=1000)
@given(
    st.one_of(
        st.binary(),
        st.text(),
        st.integers(),
        st.floats(allow_nan=False, allow_infinity=False),
        st.booleans(),
    )
)
def test_ensure_str_returns_str(value):
    result = ensure_str(value)
    assert isinstance(result, str), f"Expected str, got {type(result)}"

if __name__ == "__main__":
    test_ensure_str_returns_str()
```

<details>

<summary>
**Failing input**: `b'\x80'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/22/hypo.py", line 19, in <module>
    test_ensure_str_returns_str()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/22/hypo.py", line 5, in test_ensure_str_returns_str
    @given(

  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/22/hypo.py", line 15, in test_ensure_str_returns_str
    result = ensure_str(value)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/dtypes/common.py", line 87, in ensure_str
    value = value.decode("utf-8")
UnicodeDecodeError: 'utf-8' codec can't decode byte 0x80 in position 0: invalid start byte
Falsifying example: test_ensure_str_returns_str(
    value=b'\x80',
)
```
</details>

## Reproducing the Bug

```python
from pandas.core.dtypes.common import ensure_str

invalid_utf8_bytes = b'\x80'
result = ensure_str(invalid_utf8_bytes)
print(f"Result: {result}")
```

<details>

<summary>
UnicodeDecodeError on invalid UTF-8 byte
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/22/repo.py", line 4, in <module>
    result = ensure_str(invalid_utf8_bytes)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/dtypes/common.py", line 87, in ensure_str
    value = value.decode("utf-8")
UnicodeDecodeError: 'utf-8' codec can't decode byte 0x80 in position 0: invalid start byte
```
</details>

## Why This Is A Bug

The function's docstring states: "Ensure that bytes and non-strings get converted into `str` objects." This is an unconditional promise - there are no documented preconditions about the bytes needing to be valid UTF-8.

The function signature `ensure_str(value: bytes | Any) -> str` accepts ANY bytes object without restrictions. According to the Liskov Substitution Principle, any bytes object should be a valid input. However, the implementation assumes all bytes are UTF-8 encoded, which is not always the case in real-world data processing scenarios.

The byte value `b'\x80'` is a valid Python bytes object, but it's not valid UTF-8. The byte 0x80 by itself is an invalid UTF-8 start byte - in UTF-8, bytes from 0x80 to 0xBF can only appear as continuation bytes after a valid multi-byte start sequence.

This violates the function's contract because:
1. The type signature promises to accept any `bytes` object
2. The docstring makes no mention of encoding requirements or potential exceptions
3. The function name `ensure_str` strongly implies it will always succeed in returning a string
4. No `UnicodeDecodeError` is documented as a possible exception

## Relevant Context

This function is used internally within pandas in several critical areas:
- **pandas/core/generic.py**: Used for string matching operations in DataFrame.filter() and similar methods
- **pandas/io/json/_json.py**: Used when processing JSON data that may contain arbitrary bytes
- **pandas/core/dtypes/cast.py**: Used in type conversion operations

Real-world scenarios where this bug could occur:
- Reading data files with mixed encodings (common in legacy systems)
- Processing binary data that was incorrectly classified as text
- Handling network data with corrupted or incomplete UTF-8 sequences
- Working with data from systems using non-UTF-8 encodings (e.g., Latin-1, Windows-1252)

Python's `bytes.decode()` method supports multiple error handling strategies:
- `'strict'` (default): Raises UnicodeDecodeError
- `'replace'`: Replaces invalid bytes with � (U+FFFD)
- `'ignore'`: Skips invalid bytes
- `'backslashreplace'`: Replaces with backslash escapes

The function is located at: `/pandas/core/dtypes/common.py:82-90`

## Proposed Fix

```diff
--- a/pandas/core/dtypes/common.py
+++ b/pandas/core/dtypes/common.py
@@ -84,7 +84,7 @@ def ensure_str(value: bytes | Any) -> str:
     Ensure that bytes and non-strings get converted into ``str`` objects.
     """
     if isinstance(value, bytes):
-        value = value.decode("utf-8")
+        value = value.decode("utf-8", errors="replace")
     elif not isinstance(value, str):
         value = str(value)
     return value
```