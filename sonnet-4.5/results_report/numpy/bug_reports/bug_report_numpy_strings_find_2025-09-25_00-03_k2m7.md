# Bug Report: numpy.strings.find Returns 0 Instead of -1 for Null Character

**Target**: `numpy.strings.find`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When searching for a null character (`'\x00'`) that is not present in the string, `numpy.strings.find()` incorrectly returns 0 (found at beginning) instead of -1 (not found), violating the function's documented behavior and Python's `str.find()` semantics.

## Property-Based Test

```python
import numpy as np
import numpy.strings as nps
from hypothesis import given, strategies as st, settings

@given(st.lists(st.text(min_size=1), min_size=1).map(lambda x: np.array(x, dtype=str)),
       st.text(min_size=1, max_size=5),
       st.integers(min_value=0, max_value=20),
       st.one_of(st.integers(min_value=0, max_value=20), st.none()))
@settings(max_examples=1000)
def test_find_with_bounds(arr, sub, start, end):
    result = nps.find(arr, sub, start, end)
    for i in range(len(arr)):
        expected = arr[i].find(sub, start, end)
        assert result[i] == expected
```

**Failing input**: `arr = np.array(['abc'], dtype=str)`, `sub = '\x00'`

## Reproducing the Bug

```python
import numpy as np
import numpy.strings as nps

test_cases = ['', 'abc', 'a\x00b', '\x00']

for s in test_cases:
    arr = np.array([s], dtype=str)
    np_find = nps.find(arr, '\x00')[0]
    py_find = s.find('\x00')
    print(f"find({repr(s):8}, '\\x00'): Python={py_find:2}, NumPy={np_find:2}")
```

Output:
```
find('':8, '\x00'): Python=-1, NumPy= 0
find('abc':8, '\x00'): Python=-1, NumPy= 0
find('a\x00b':8, '\x00'): Python= 1, NumPy= 0
find('\x00':8, '\x00'): Python= 0, NumPy= 0
```

## Why This Is A Bug

1. **Violates API contract**: The function is documented to return -1 when substring is not found, but returns 0 instead for null character searches.

2. **Inconsistent with Python**: Python's `str.find('\x00')` correctly returns -1 when the null character is not present.

3. **Wrong even when found**: For `'a\x00b'.find('\x00')`, Python correctly returns 1 (the actual position), but NumPy returns 0 (incorrect).

4. **Cannot distinguish found/not-found**: Users cannot tell if 0 means "found at position 0" or "not found but returned wrong value".

## Fix

The implementation is treating null character searches specially, likely as zero-width patterns that match at position 0. The search logic should handle null characters as regular single-byte characters and properly return -1 when not found.