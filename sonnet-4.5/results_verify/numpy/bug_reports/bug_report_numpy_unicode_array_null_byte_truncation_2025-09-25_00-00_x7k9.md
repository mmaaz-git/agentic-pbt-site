# Bug Report: NumPy Unicode Array - Trailing Null Byte Truncation

**Target**: `numpy.array` (affects `numpy.strings` module)
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25
**NumPy Version**: 2.3.0

## Summary

NumPy silently truncates trailing null bytes when creating Unicode string arrays, causing data corruption for any string ending with '\x00'.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st, example


@given(st.text())
@example('\x00')
@example('test\x00')
@example('end\x00\x00')
def test_array_preserves_null_bytes(s):
    arr = np.array([s])
    result = str(arr[0])
    assert result == s, f"Input {s!r} became {result!r}"
```

**Failing inputs**:
- `'\x00'` → becomes `''`
- `'test\x00'` → becomes `'test'`
- `'end\x00\x00'` → becomes `'end'`

## Reproducing the Bug

```python
import numpy as np

test_cases = [
    '\x00',
    'end\x00',
    'hello\x00world',
    '\x00start',
    'test\x00\x00',
]

for s in test_cases:
    arr = np.array([s])
    result = str(arr[0])
    match = result == s
    print(f"Input: {s!r:20s} | Array: {result!r:20s} | Match: {match}")
```

Output:
```
Input: '\x00'               | Array: ''                   | Match: False
Input: 'end\x00'            | Array: 'end'                | Match: False
Input: 'hello\x00world'     | Array: 'hello\x00world'     | Match: True
Input: '\x00start'          | Array: '\x00start'          | Match: True
Input: 'test\x00\x00'       | Array: 'test'               | Match: False
```

## Why This Is A Bug

This violates the fundamental expectation that creating an array from data should preserve that data. The behavior is:
- **Trailing null bytes**: Silently removed
- **Strings of only null bytes**: Become empty strings
- **Leading/interior null bytes**: Preserved correctly

This silent data corruption can lead to serious issues in applications that:
- Process binary data represented as strings
- Use null bytes as delimiters or markers
- Work with C-style strings or serialization formats
- Perform any null-byte-aware string processing

The bug affects all `numpy.strings` module functions since they operate on NumPy arrays.

## Fix

This appears to be a C-level issue in NumPy's Unicode string handling, likely related to C-string null termination assumptions. The Unicode dtype implementation may be treating null bytes as string terminators.

Workaround: Use `dtype=object` for strings containing trailing null bytes:
```python
arr = np.array([s], dtype=object)
```

A proper fix would require modifying NumPy's internal Unicode string handling to correctly store and retrieve strings with trailing null bytes, ensuring the length field is used rather than relying on null termination.