# Bug Report: numpy.char.array and asarray Strip Whitespace Characters

**Target**: `numpy.char.array`, `numpy.char.asarray`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`numpy.char.array()` and `numpy.char.asarray()` silently strip whitespace characters including `\r`, `\n`, `\t`, and `\x00` from input strings, causing data loss.

## Property-Based Test

```python
import numpy as np
import numpy.char
from hypothesis import given, strategies as st


@given(st.lists(st.text(), min_size=1))
def test_chararray_preserves_data(strings):
    arr = numpy.char.array(strings)

    for i in range(len(strings)):
        assert str(arr[i]) == strings[i]
```

**Failing input**: `strings=['\r']`

## Reproducing the Bug

```python
import numpy as np
import numpy.char

test_cases = ['\r', '\n', '\t', '\x00', 'hello\r', 'world\n']

for test_string in test_cases:
    arr = numpy.char.array([test_string])
    print(f"Input:  {repr(test_string)}")
    print(f"Output: {repr(str(arr[0]))}")
    print(f"Match:  {str(arr[0]) == test_string}")
    print()
```

Output:
```
Input:  '\r'
Output: ''
Match:  False

Input:  '\n'
Output: ''
Match:  False

Input:  '\t'
Output: ''
Match:  False

Input:  '\x00'
Output: ''
Match:  False

Input:  'hello\r'
Output: 'hello'
Match:  False

Input:  'world\n'
Output: 'world'
Match:  False
```

## Why This Is A Bug

1. **Silent data corruption**: Whitespace characters are silently removed without warning
2. **No documentation**: This behavior is not documented in the docstring
3. **Unexpected behavior**: Users expect array creation to preserve their data
4. **Affects real use cases**: Log parsing, text processing, CSV data, etc. all use these characters

## Fix

The bug likely stems from internal C string handling treating these as terminators or trim characters. The fix would require:

1. Identify where in the chararray creation code whitespace is being stripped
2. Remove or make optional the stripping behavior
3. Ensure proper handling of all Unicode whitespace and control characters