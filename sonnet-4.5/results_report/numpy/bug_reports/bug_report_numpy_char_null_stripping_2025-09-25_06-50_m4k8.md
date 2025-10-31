# Bug Report: numpy.char Case Transformations Strip Null Bytes

**Target**: `numpy.char.upper`, `numpy.char.lower`, `numpy.char.capitalize`, `numpy.char.title`, `numpy.char.swapcase`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

Case transformation functions in numpy.char silently remove null bytes (`\x00`) from strings, causing data corruption. Python's str methods preserve null bytes, but numpy.char's implementations do not.

## Property-Based Test

```python
import numpy.char as char
from hypothesis import given, strategies as st, settings

@given(st.text(min_size=0, max_size=20))
@settings(max_examples=1000)
def test_transformations_preserve_null_bytes(s):
    transformations = [
        ('upper', char.upper),
        ('lower', char.lower),
        ('capitalize', char.capitalize),
        ('title', char.title),
        ('swapcase', char.swapcase),
    ]

    null_count_original = s.count('\x00')

    for name, func in transformations:
        result = str(func(s))
        null_count_result = result.count('\x00')

        if null_count_result < null_count_original:
            raise AssertionError(
                f"{name}('{repr(s)}') lost null bytes: "
                f"{null_count_original} -> {null_count_result}"
            )
```

**Failing input**: Any string containing `\x00`, e.g., `s='\x00'`

## Reproducing the Bug

```python
import numpy.char as char

s = '\x00'

print(f"Original: {repr(s)} (length={len(s)})")
print(f"Python upper: {repr(s.upper())} (length={len(s.upper())})")
print(f"NumPy upper:  {repr(str(char.upper(s)))} (length={len(str(char.upper(s)))})")

s2 = 'te\x00st'
print(f"\nOriginal: {repr(s2)} (length={len(s2)})")
print(f"Python lower: {repr(s2.lower())} (length={len(s2.lower())})")
print(f"NumPy lower:  {repr(str(char.lower(s2)))} (length={len(str(char.lower(s2)))})")
```

Output:
```
Original: '\x00' (length=1)
Python upper: '\x00' (length=1)
NumPy upper:  '' (length=0)

Original: 'te\x00st' (length=6)
Python lower: 'te\x00st' (length=6)
NumPy lower:  'test' (length=4)
```

## Why This Is A Bug

1. **Silent data corruption**: Null bytes are valid string characters in Python strings. Silently removing them corrupts user data.

2. **Violates documented contract**: The functions claim to call str methods element-wise, but they produce different results than Python's str methods for strings containing null bytes.

3. **Security implications**: Null bytes are used in many security-sensitive contexts (null-terminated strings from C, binary protocols, etc.). Silently removing them could create security vulnerabilities.

4. **Unexpected behavior**: Users expect case transformations to preserve string content, only changing case. Removing characters violates this expectation.

## Fix

The case transformation functions likely use a C implementation that treats null bytes as string terminators. They should be updated to handle null bytes correctly, either by:

1. Using Python's unicode string operations that handle null bytes properly
2. Explicitly preserving null bytes in the C implementation
3. Documenting this limitation if it cannot be fixed

The simplest fix would be to ensure the underlying implementation treats the string as a fixed-length buffer rather than null-terminated.