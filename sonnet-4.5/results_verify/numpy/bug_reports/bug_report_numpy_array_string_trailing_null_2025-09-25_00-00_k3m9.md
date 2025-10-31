# Bug Report: NumPy String Arrays Truncate at Trailing Null Characters

**Target**: `numpy.array` (string dtype)
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

NumPy string arrays silently truncate strings at trailing null characters (`\x00`), causing data loss. Null characters in the middle of strings are preserved, but trailing nulls are removed, which violates Python's string semantics.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import numpy as np

@given(st.lists(st.text(min_size=1, max_size=20), min_size=1, max_size=50))
@settings(max_examples=500)
def test_char_array_length_consistency(strings):
    arr = np.array(strings, dtype=str)

    for i, s in enumerate(strings):
        assert arr[i] == s, f"string array doesn't preserve strings"
```

**Failing input**: `strings=['\x00']` (and any string ending with `\x00`)

## Reproducing the Bug

```python
import numpy as np

s1 = 'world\x00'
arr1 = np.array([s1], dtype=str)
print(f"Input: {repr(s1)}, length: {len(s1)}")
print(f"Stored: {repr(arr1[0])}, length: {len(arr1[0])}")
assert s1 == arr1[0]
```

Output:
```
Input: 'world\x00', length: 6
Stored: 'world', length: 5
AssertionError
```

Additional test cases demonstrating the pattern:
```python
import numpy as np

test_cases = [
    ('\x00', ''),           # Only null -> empty string
    ('a\x00', 'a'),         # Trailing null removed
    ('\x00b', '\x00b'),     # Leading null preserved
    ('a\x00b', 'a\x00b'),   # Middle null preserved
    ('world\x00', 'world'), # Trailing null removed
]

for input_str, expected_output in test_cases:
    arr = np.array([input_str], dtype=str)
    actual = arr[0]
    print(f"{repr(input_str):15} -> {repr(actual):15} (expected: {repr(expected_output)})")
    assert actual == expected_output
```

## Why This Is A Bug

Python strings can contain null bytes (`\x00`) at any position, including at the end. NumPy string arrays should preserve the exact string values, but instead they truncate at trailing null characters, treating them as C-style string terminators. This causes silent data loss and violates the documented behavior that arrays preserve their input values.

The inconsistency is particularly problematic: null characters in the middle of strings are preserved correctly, but trailing nulls are silently removed. Users have no way to detect this data loss without explicitly checking.

## Fix

This bug likely stems from NumPy's internal use of C strings or Unicode string handling that treats null as a terminator. The fix would need to:

1. Ensure that string length is determined from Python string's `len()`, not from searching for null terminators
2. Preserve all bytes in the string, including trailing nulls
3. Add tests to verify that strings with nulls at any position (start, middle, end) are preserved exactly

A potential workaround for users is to use object dtype instead of string dtype, though this defeats the purpose of typed string arrays:
```python
arr = np.array(['world\x00'], dtype=object)
```