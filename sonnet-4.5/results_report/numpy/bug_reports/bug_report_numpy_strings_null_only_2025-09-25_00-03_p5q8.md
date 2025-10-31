# Bug Report: numpy.strings upper/lower Return Empty String for Null-Only Strings

**Target**: `numpy.strings.upper`, `numpy.strings.lower`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`numpy.strings.upper()` and `numpy.strings.lower()` return empty string for strings consisting only of null characters, instead of preserving them unchanged.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy as np
import numpy.strings as nps

@given(st.lists(st.text(), min_size=1, max_size=10))
def test_upper_matches_python(strings):
    for s in strings:
        arr = np.array([s])
        np_result = nps.upper(arr)[0]
        py_result = s.upper()
        assert np_result == py_result
```

**Failing input**: `strings=['\x00']`

## Reproducing the Bug

```python
import numpy as np
import numpy.strings as nps

test_cases = ['\x00', '\x00\x00', 'a\x00b', 'hello\x00world']

for test_str in test_cases:
    arr = np.array([test_str])
    py_upper = test_str.upper()
    np_upper = nps.upper(arr)[0]
    py_lower = test_str.lower()
    np_lower = nps.lower(arr)[0]

    print(f"Input: {repr(test_str)}")
    print(f"  upper() - Python: {repr(py_upper)}, NumPy: {repr(np_upper)}")
    print(f"  lower() - Python: {repr(py_lower)}, NumPy: {repr(np_lower)}")
```

## Why This Is A Bug

1. Null character is a valid Unicode character that should be preserved
2. Python's `str.upper()` and `str.lower()` correctly return `'\x00'` unchanged
3. NumPy returns empty string for `'\x00'` and `'\x00\x00'`
4. Bug only affects strings that consist ONLY of null characters
5. Mixed strings like `'a\x00b'` are handled correctly
6. Violates documented behavior of calling str methods element-wise

## Fix

The implementation likely uses C string operations that treat `\x00` as string terminator. The fix requires using length-aware string operations that don't rely on null termination, or special handling to preserve null characters in the output buffer.