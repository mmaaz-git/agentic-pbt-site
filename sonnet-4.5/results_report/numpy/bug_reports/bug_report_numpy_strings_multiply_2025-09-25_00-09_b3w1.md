# Bug Report: numpy.strings.multiply Returns Empty String for Null Characters

**Target**: `numpy.strings.multiply` (string repetition)
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`numpy.strings.multiply()` incorrectly returns an empty string when multiplying strings that consist only of null characters, instead of repeating them as expected.

## Property-Based Test

```python
import numpy as np
import numpy.strings as nps
from hypothesis import given, strategies as st, settings

@given(st.lists(st.text(min_size=1, max_size=20), min_size=1, max_size=5))
@settings(max_examples=500)
def test_multiply_broadcast(strings):
    arr = np.array(strings, dtype=str)
    n = 3
    result = nps.multiply(arr, n)
    for i in range(len(arr)):
        expected = strings[i] * n
        assert result[i] == expected
```

**Failing input**: `strings = ['\x00']`

## Reproducing the Bug

```python
import numpy as np
import numpy.strings as nps

test_cases = [
    ('\x00', 3),
    ('\x00\x00', 2),
    ('a\x00', 2),
]

for s, n in test_cases:
    arr = np.array([s], dtype=str)
    result = nps.multiply(arr, n)[0]
    expected = s * n
    print(f"multiply({repr(s):10}, {n}): Expected={repr(expected):20}, Got={repr(result):20}")
```

Output:
```
multiply('\x00'     , 3): Expected='\x00\x00\x00'       , Got=''
multiply('\x00\x00' , 2): Expected='\x00\x00\x00\x00'   , Got=''
multiply('a\x00'    , 2): Expected='a\x00a\x00'         , Got='a\x00a\x00'
```

## Why This Is A Bug

1. **Completely wrong result**: Returns empty string instead of repeating null characters.

2. **Inconsistent with Python**: Python's string repetition `'\x00' * 3` correctly produces `'\x00\x00\x00'`.

3. **Partial failures**: Works correctly when the string contains non-null characters mixed with nulls, but fails for pure null character strings.

4. **Violates length invariant**: The documented behavior states that multiplying should produce a string of length `len(s) * n`, but this returns length 0 instead.

## Fix

The implementation likely treats strings starting with null characters as empty (zero-length), similar to how C treats null-terminated strings. The string repetition logic should use the actual string length rather than stopping at the first null character.