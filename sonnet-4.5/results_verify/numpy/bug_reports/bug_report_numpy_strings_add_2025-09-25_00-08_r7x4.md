# Bug Report: numpy.strings.add Strips Leading Null Characters

**Target**: `numpy.strings.add` (string concatenation)
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`numpy.strings.add()` (string concatenation) incorrectly strips leading null characters from the first operand during concatenation, causing data loss.

## Property-Based Test

```python
import numpy as np
import numpy.strings as nps
from hypothesis import given, strategies as st, settings

@given(st.lists(st.text(), min_size=1, max_size=5))
@settings(max_examples=500)
def test_add_broadcast(strings):
    arr = np.array(strings, dtype=str)
    scalar = 'test'
    result = nps.add(arr, scalar)
    for i in range(len(arr)):
        expected = strings[i] + scalar
        assert result[i] == expected
```

**Failing input**: `strings = ['\x00']`

## Reproducing the Bug

```python
import numpy as np
import numpy.strings as nps

test_cases = [
    ('\x00', 'test'),
    ('\x00\x00', 'abc'),
    ('a\x00', 'b'),
]

for s1, s2 in test_cases:
    arr1 = np.array([s1], dtype=str)
    result = nps.add(arr1, s2)[0]
    expected = s1 + s2
    print(f"add({repr(s1):10}, {repr(s2):8}): Expected={repr(expected):15}, Got={repr(result):15}")
```

Output:
```
add('\x00'     , 'test'  ): Expected='\x00test'       , Got='test'
add('\x00\x00' , 'abc'   ): Expected='\x00\x00abc'    , Got='abc'
add('a\x00'    , 'b'     ): Expected='a\x00b'         , Got='a\x00b'
```

## Why This Is A Bug

1. **Data loss**: Leading null characters are silently stripped from the first operand.

2. **Inconsistent with Python**: Python's string concatenation `'\x00' + 'test'` correctly produces `'\x00test'`.

3. **Not consistent with all positions**: Trailing nulls in the first operand are preserved (`'a\x00' + 'b'` works correctly), but leading nulls are stripped.

4. **Breaks associativity**: String concatenation should be associative, but this bug breaks that property.

## Fix

The concatenation implementation is likely treating the first string as a null-terminated C string, stopping at the first null character when copying. It should use length-aware string copying that preserves all characters including null bytes.