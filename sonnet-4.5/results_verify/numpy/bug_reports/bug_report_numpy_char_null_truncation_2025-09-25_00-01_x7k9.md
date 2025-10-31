# Bug Report: NumPy Unicode String Null Character Truncation

**Target**: `numpy.array` (core array creation), affects `numpy.char` functions
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

NumPy silently truncates Unicode strings containing trailing null characters (`\x00`) when creating arrays. This causes data loss and affects all `numpy.char` string manipulation functions.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import numpy as np
import numpy.char as char

@given(st.text(max_size=50))
@settings(max_examples=500)
def test_case_preserves_length(s):
    upper_result = char.upper(s)
    upper_str = str(upper_result)
    assert len(upper_str) == len(s), f"upper changed length: {len(s)} -> {len(upper_str)}"
```

**Failing input**: `'\x00'` (and any string ending with `\x00`)

## Reproducing the Bug

```python
import numpy as np

s1 = '\x00'
arr1 = np.array(s1)
print(f"Input:  {repr(s1)} (length={len(s1)})")
print(f"Output: {repr(str(arr1))} (length={len(str(arr1))})")

s2 = 'abc\x00'
arr2 = np.array(s2)
print(f"Input:  {repr(s2)} (length={len(s2)})")
print(f"Output: {repr(str(arr2))} (length={len(str(arr2))})")

s3 = '\x00abc'
arr3 = np.array(s3)
print(f"Input:  {repr(s3)} (length={len(s3)})")
print(f"Output: {repr(str(arr3))} (length={len(str(arr3))})")
```

**Expected output:**
```
Input:  '\x00' (length=1)
Output: '\x00' (length=1)
Input:  'abc\x00' (length=4)
Output: 'abc\x00' (length=4)
Input:  '\x00abc' (length=4)
Output: '\x00abc' (length=4)
```

**Actual output:**
```
Input:  '\x00' (length=1)
Output: '' (length=0)
Input:  'abc\x00' (length=4)
Output: 'abc' (length=3)
Input:  '\x00abc' (length=4)
Output: '\x00abc' (length=4)
```

## Why This Is A Bug

1. **Silent data corruption**: Characters are removed without warning or error
2. **Violates Python string semantics**: Python strings can contain null characters; `'\x00'.upper() == '\x00'` works correctly
3. **Inconsistent behavior**: Null characters in the middle are preserved, but trailing nulls are truncated
4. **C-string contamination**: NumPy appears to treat Unicode strings as C-style null-terminated strings, which is incorrect for Python

The pattern shows:
- `'\x00'` → `''` (leading null removed when it's the only character)
- `'a\x00'` → `'a'` (trailing null removed)
- `'\x00a'` → `'\x00a'` (leading null preserved if followed by other chars)
- `'a\x00b'` → `'a\x00b'` (middle null preserved)

This suggests the bug is in how NumPy copies/converts strings to its internal representation.

## Fix

The issue is likely in NumPy's Unicode string handling in `numpy/_core/src/multiarray/dtypes.c` or similar C code that implements the Unicode dtype. The code is probably using C string functions that treat `\x00` as a terminator rather than properly handling Python's length-prefixed strings.

A proper fix would require:
1. Ensuring all string length calculations use the Python string's actual length, not C's `strlen`
2. Using memory copy operations that respect the full length rather than stopping at null
3. Adding tests for null character handling in string arrays

Since this is a fundamental issue in NumPy's C implementation and requires deep knowledge of the codebase, I cannot provide a simple patch. The NumPy team would need to audit their Unicode string handling code to ensure it properly handles embedded and trailing null characters.