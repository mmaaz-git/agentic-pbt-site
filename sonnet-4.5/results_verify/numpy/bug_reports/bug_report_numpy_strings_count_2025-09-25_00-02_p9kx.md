# Bug Report: numpy.strings.count Treats Null Character as Matching Everywhere

**Target**: `numpy.strings.count`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When counting occurrences of a null character (`'\x00'`), `numpy.strings.count()` incorrectly treats it as matching between every character in the string, returning `len(s) + 1` instead of the actual count of null characters.

## Property-Based Test

```python
import numpy as np
import numpy.strings as nps
from hypothesis import given, strategies as st, settings

@given(st.lists(st.text(), min_size=1, max_size=10).map(lambda x: np.array(x, dtype=str)),
       st.text(min_size=1, max_size=5),
       st.integers(min_value=0, max_value=20),
       st.one_of(st.integers(min_value=0, max_value=20), st.none()))
@settings(max_examples=1000)
def test_count_with_bounds(arr, sub, start, end):
    result = nps.count(arr, sub, start, end)
    for i in range(len(arr)):
        expected = arr[i].count(sub, start, end)
        assert result[i] == expected
```

**Failing input**: `arr = np.array(['abc'], dtype=str)`, `sub = '\x00'`

## Reproducing the Bug

```python
import numpy as np
import numpy.strings as nps

test_cases = [
    '',
    'abc',
    'a\x00b',
    '\x00\x00',
]

for s in test_cases:
    arr = np.array([s], dtype=str)
    np_count = nps.count(arr, '\x00')[0]
    py_count = s.count('\x00')
    print(f"count({repr(s):10}, '\\x00'): Python={py_count}, NumPy={np_count}")
```

Output:
```
count('':10, '\x00'): Python=0, NumPy=1
count('abc':10, '\x00'): Python=0, NumPy=4
count('a\x00b':10, '\x00'): Python=1, NumPy=4
count('\x00\x00':10, '\x00'): Python=2, NumPy=1
```

## Why This Is A Bug

1. **Incorrect results**: Returns the wrong count for strings without null characters (returns `len(s) + 1` instead of 0).

2. **Inconsistent with Python**: Python's `str.count('\x00')` correctly returns the actual number of null characters.

3. **Unpredictable behavior**: For `'\x00\x00'.count('\x00')`, returns 1 instead of 2, showing the pattern-matching logic is fundamentally broken for this case.

4. **Data integrity**: Code relying on count for validation or parsing will produce incorrect results when null characters are involved.

## Fix

The C implementation likely treats null characters as zero-width patterns or string terminators. The search/match logic needs to properly handle null characters as regular single-byte characters to search for, not as special markers.