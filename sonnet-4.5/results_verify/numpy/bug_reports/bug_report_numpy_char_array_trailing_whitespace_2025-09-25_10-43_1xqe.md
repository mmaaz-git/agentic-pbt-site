# Bug Report: numpy.char.array Trailing Whitespace and Null Byte Truncation

**Target**: `numpy.char.array`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`numpy.char.array` silently strips trailing whitespace characters and truncates trailing null bytes, violating the contract that it should faithfully represent Python strings element-wise. Whitespace-only strings become empty strings.

## Property-Based Test

```python
import numpy.char as char
from hypothesis import given, strategies as st, settings


@given(st.text())
@settings(max_examples=500)
def test_swapcase_involution(s):
    arr = char.array([s])
    result = char.swapcase(char.swapcase(arr))
    assert result[0] == s
```

**Failing input**: `'\x00'`

## Reproducing the Bug

```python
import numpy.char as char

test_cases = [
    ' ',
    '  ',
    '\t',
    '\n',
    '\r',
    '\x00',
    'end\x00',
]

for s in test_cases:
    arr = char.array([s])
    print(f"Input: {repr(s):<10} len={len(s)}  ->  Output: {repr(arr[0]):<10} len={len(arr[0])}  Match: {arr[0] == s}")
```

Output:
```
Input: ' '        len=1  ->  Output: ''         len=0  Match: False
Input: '  '       len=2  ->  Output: ''         len=0  Match: False
Input: '\t'       len=1  ->  Output: ''         len=0  Match: False
Input: '\n'       len=1  ->  Output: ''         len=0  Match: False
Input: '\r'       len=1  ->  Output: ''         len=0  Match: False
Input: '\x00'     len=1  ->  Output: ''         len=0  Match: False
Input: 'end\x00'  len=4  ->  Output: 'end'      len=3  Match: False
```

## Why This Is A Bug

Python strings can contain whitespace and null bytes, and these characters should be preserved. NumPy's `char.array` appears to be stripping trailing whitespace and treating null bytes as C-style string terminators. This violates:

1. The documented behavior that these are "element-wise" operations matching Python string methods
2. Python's string semantics where whitespace and null bytes are valid characters
3. User expectations that string data won't be silently corrupted
4. The principle of least surprise - users expect arrays to faithfully store their input

This is particularly severe because:
- It causes **silent data corruption** with no warning
- Whitespace-only strings (spaces, tabs, newlines) become empty strings
- Any trailing whitespace is stripped, changing string values
- Affects all downstream operations (`swapcase`, `add`, `multiply`, etc.)

## Fix

This is likely a lower-level issue in NumPy's string handling. The fix would require changing how NumPy's string dtype stores strings internally to not use C-style null-terminated strings, or to properly escape null bytes. This is a non-trivial architectural change that would require investigation into NumPy's core string implementation.