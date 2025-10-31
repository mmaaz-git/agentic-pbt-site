# Bug Report: numpy.strings.replace Truncates When Whole String Expands

**Target**: `numpy.strings.replace`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`numpy.strings.replace()` silently truncates output when replacing the entire string with a longer replacement, returning the original string instead of the expected replacement.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy as np
import numpy.strings as nps

@given(st.lists(st.text(min_size=1), min_size=1, max_size=10), st.text(min_size=1, max_size=5), st.text(max_size=5))
def test_replace_matches_python(strings, old, new):
    for s in strings:
        if old in s:
            arr = np.array([s])
            np_result = nps.replace(arr, old, new)[0]
            py_result = s.replace(old, new)
            assert np_result == py_result
```

**Failing input**: `strings=['0']`, `old='0'`, `new='00'`

## Reproducing the Bug

```python
import numpy as np
import numpy.strings as nps

test_cases = [
    ('0', '0', '00'),
    ('a', 'a', 'aa'),
    ('hello', 'hello', 'hellohello'),
    ('hello', 'l', 'll'),
]

for s, old, new in test_cases:
    arr = np.array([s])
    py_result = s.replace(old, new)
    np_result = nps.replace(arr, old, new)[0]
    match = 'PASS' if py_result == np_result else 'FAIL'
    print(f"{match}: replace('{s}', '{old}', '{new}')")
    print(f"  Expected: '{py_result}'")
    print(f"  Actual: '{np_result}'")
```

## Why This Is A Bug

1. When replacing entire string with longer replacement, NumPy returns original unchanged
2. Bug only occurs when: `s == old` AND `len(new) > len(old)`
3. Partial replacements work correctly: `replace('hello', 'l', 'll')` correctly returns `'hellllo'`
4. Violates documented behavior of calling `str.replace()` element-wise
5. Causes silent data corruption - operation appears to succeed but produces wrong result

## Fix

The implementation uses a fixed-size output buffer based on input string length. When the entire string is replaced with a longer string, the buffer is too small and the operation silently fails, returning the original. The fix requires proper buffer size calculation that accounts for expansion during replacement.