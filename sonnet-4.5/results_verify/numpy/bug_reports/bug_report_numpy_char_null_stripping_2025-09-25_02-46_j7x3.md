# Bug Report: numpy.char Functions Strip Null Characters

**Target**: `numpy.char` (all string transformation functions)
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

All string transformation functions in `numpy.char` silently strip null characters (`\x00`) from strings, causing silent data corruption. Affected functions include `upper()`, `lower()`, `capitalize()`, `title()`, `swapcase()`, `strip()`, `lstrip()`, `rstrip()`, and `encode()`.

## Property-Based Test

```python
import numpy.char as char
from hypothesis import given, strategies as st, settings

@given(st.text(min_size=0, max_size=100))
@settings(max_examples=500, deadline=None)
def test_encode_decode_roundtrip(s):
    encoded = char.encode(s, encoding='utf-8')
    decoded = char.decode(encoded, encoding='utf-8')
    decoded_str = str(decoded) if hasattr(decoded, 'item') else decoded
    assert decoded_str == s
```

**Failing input**: `s='\x00'`

## Reproducing the Bug

```python
import numpy.char as char

s = '\x00'

functions = ['upper', 'lower', 'capitalize', 'title', 'swapcase', 'strip']
for func_name in functions:
    numpy_func = getattr(char, func_name)
    python_func = getattr(str, func_name)

    numpy_result = numpy_func(s).item()
    python_result = python_func(s)

    print(f"{func_name}({repr(s)}): numpy={repr(numpy_result)}, python={repr(python_result)}")
    assert numpy_result == python_result
```

Expected: All functions preserve the null character
Actual: All functions return empty string `''`

## Why This Is A Bug

1. **Silent data corruption**: Null characters are silently removed without warning
2. **Violates documented behavior**: Docstrings claim to call str methods "element-wise", but behavior differs from Python's str methods
3. **Breaks fundamental assumptions**: Users expect string operations to preserve all valid Unicode characters
4. **Affects binary data**: Null bytes in text data (e.g., from binary protocols) are legitimate and should be preserved

## Fix

This bug likely originates from numpy treating null bytes as C-style string terminators. The fix requires:

1. Ensuring internal string handling doesn't treat `\x00` as a terminator
2. Properly handling null bytes in the underlying string ufunc implementations
3. Testing all string operations with strings containing null characters

The implementation is in `numpy/_core/defchararray.py` and the underlying C/Cython code that handles string operations.