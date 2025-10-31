# Bug Report: numpy.char Case Operations Strip Null Characters

**Target**: `numpy.char.upper`, `numpy.char.lower`, `numpy.char.swapcase`, `numpy.char.title`, `numpy.char.capitalize`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

Case transformation operations silently strip null characters (`\x00`) from strings, causing data corruption.

## Property-Based Test

```python
import numpy as np
import numpy.char as char
from hypothesis import given, settings, strategies as st


@settings(max_examples=200)
@given(st.text(alphabet='\x00abcABC', min_size=1, max_size=10))
def test_upper_preserves_null_chars(s):
    arr = np.array([s], dtype='U100')
    result = char.upper(arr)

    expected = s.upper()
    actual = str(result[0])

    assert actual == expected, f"Input: {s!r}, Expected: {expected!r}, Got: {actual!r}"
```

**Failing input**: `s='\x00'`

## Reproducing the Bug

```python
import numpy as np
import numpy.char as char

arr = np.array(['\x00'], dtype='U100')
result = char.upper(arr)

print(f"Input: {arr[0]!r}, len={len(arr[0])}")
print(f"Result: {result[0]!r}, len={len(str(result[0]))}")
print(f"Python behavior: {'\x00'.upper()!r}")

assert str(result[0]) == '\x00', f"Expected '\\x00', got {result[0]!r}"
```

Output:
```
Input: np.str_(''), len=0
Result: np.str_(''), len=0
Python behavior: '\x00'
AssertionError: Expected '\x00', got np.str_('')
```

## Why This Is A Bug

1. Null characters (`\x00`) are valid in Python strings
2. Python's `str.upper()` correctly preserves null characters
3. numpy.char.upper (and other case operations) silently strips them, returning empty strings
4. This violates the documented behavior "Calls str.upper element-wise"
5. Users may have binary data or protocol strings with embedded nulls that get silently corrupted

## Fix

The underlying C implementation likely uses null-terminated strings, causing premature string termination. The fix requires updating the C implementation to properly handle strings with embedded null characters, or explicitly documenting this limitation.

```diff
# Conceptual fix in C code (actual implementation)
- // Use null-terminated string operations
- char *result = toupper_c_string(input);  // stops at \x00
+ // Use length-aware string operations
+ char *result = toupper_with_length(input, length);  // processes full length
```