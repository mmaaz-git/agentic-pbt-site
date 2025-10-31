# Bug Report: pandas.core.dtypes.common.ensure_str Invalid UTF-8 Decode

**Target**: `pandas.core.dtypes.common.ensure_str`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `ensure_str` function crashes when given bytes that are not valid UTF-8, violating its documented behavior of ensuring "bytes and non-strings get converted into str objects."

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
```

**Failing input**: `b'\x80'` (byte 0x80 is not valid UTF-8)

## Reproducing the Bug

```python
from pandas.core.dtypes.common import ensure_str

invalid_utf8_bytes = b'\x80'
result = ensure_str(invalid_utf8_bytes)
```

Output:
```
UnicodeDecodeError: 'utf-8' codec can't decode byte 0x80 in position 0: invalid start byte
```

## Why This Is A Bug

The function's docstring states: "Ensure that bytes and non-strings get converted into `str` objects."

However, the function crashes on bytes that contain invalid UTF-8 sequences. According to its contract, it should handle all bytes objects, not just valid UTF-8 bytes. The function currently has no documented preconditions about the bytes needing to be valid UTF-8.

This is a crash bug because:
1. The type signature accepts any `bytes` object
2. The docstring promises conversion without preconditions
3. The function raises an unexpected `UnicodeDecodeError` instead of converting or raising a documented error

## Fix

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

This fix uses `errors="replace"` to handle invalid UTF-8 bytes gracefully by replacing them with the Unicode replacement character (�), ensuring the function always returns a string as documented.