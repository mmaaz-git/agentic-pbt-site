# Bug Report: numpy.char Trailing Null Character Truncation

**Target**: `numpy.char` (multiple functions including `multiply`, but affects array creation)
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When creating numpy string arrays, trailing null characters (`\x00`) are silently truncated, causing data loss. This affects all numpy.char operations and violates the expected behavior that strings should be preserved intact.

## Property-Based Test

```python
import numpy as np
import numpy.char
from hypothesis import given, strategies as st

@given(st.lists(st.text(), min_size=1), st.integers(min_value=0, max_value=100))
def test_multiply_length_property(strings, n):
    arr = np.array(strings)
    result = numpy.char.multiply(arr, n)
    for i, s in enumerate(strings):
        assert len(result[i]) == len(s) * n
```

**Failing input**: `strings=['\x00'], n=1`

## Reproducing the Bug

```python
import numpy as np
import numpy.char

s = 'test\x00'
arr = np.array([s])
result = numpy.char.multiply(arr, 1)

print(f'Input: {s!r} (len={len(s)})')
print(f'Expected: {s * 1!r} (len={len(s * 1)})')
print(f'Actual: {result[0]!r} (len={len(result[0])})')

assert result[0] == s
```

**Output**:
```
Input: 'test\x00' (len=5)
Expected: 'test\x00' (len=5)
Actual: 'test' (len=4)
AssertionError
```

## Why This Is A Bug

1. Trailing null characters are valid in Python strings and should be preserved
2. The truncation happens silently without warning or error
3. This causes data loss and violates the documented behavior that operations should work "element-wise" like Python's string methods
4. Python's `str * n` preserves null characters: `'test\x00' * 1 == 'test\x00'`
5. The bug appears to be in numpy's string array creation/handling, not specific to `multiply`

This is particularly problematic for:
- Binary string data that may contain null bytes
- C-string interop where null terminators are significant
- Any application requiring exact string preservation

## Fix

The issue is in numpy's fixed-width Unicode string dtype (`<U` dtype), which appears to be treating null characters as C-string terminators. The root cause is likely in numpy's core string array implementation, not in numpy.char specifically.

Workaround: Use `dtype=object` for strings containing null characters:
```python
arr = np.array(['test\x00'], dtype=object)
```

A proper fix would require modifying numpy's Unicode string handling to not treat `\x00` as a terminator for Python strings.