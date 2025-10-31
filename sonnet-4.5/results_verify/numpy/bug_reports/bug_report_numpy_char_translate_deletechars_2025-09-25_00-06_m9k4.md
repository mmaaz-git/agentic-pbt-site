# Bug Report: numpy.char.translate deletechars Parameter Non-Functional for Unicode Strings

**Target**: `numpy.char.translate`
**Severity**: High
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `deletechars` parameter in `numpy.char.translate` is documented and present in the function signature, but completely non-functional for Unicode strings (str dtype). It works correctly for bytes arrays but has no effect on string arrays.

## Property-Based Test

```python
import numpy as np
import numpy.char as char
from hypothesis import given, strategies as st

@given(st.lists(st.text(alphabet='ABCXYZ', min_size=1, max_size=10), min_size=1, max_size=10),
       st.text(alphabet='ABC', min_size=1, max_size=3))
def test_translate_deletechars_removes_chars(strings, deletechars):
    arr = np.array(strings, dtype=str)
    table = str.maketrans('', '')
    result = char.translate(arr, table, deletechars=deletechars)

    for i in range(len(strings)):
        expected = ''.join(c for c in strings[i] if c not in deletechars)
        assert result[i] == expected
```

**Failing input**: `strings=['A'], deletechars='A'`

## Reproducing the Bug

```python
import numpy as np
import numpy.char as char

arr = np.array(['ABCDEF'], dtype=str)
table = str.maketrans('', '')
result = char.translate(arr, table, deletechars='A')

print(f"Result: {result[0]!r}")
print(f"Expected: 'BCDEF'")
assert result[0] == 'BCDEF'
```

## Why This Is A Bug

The function signature includes a `deletechars` parameter, and the documentation describes it as "characters to be removed". However, the parameter has no effect for Unicode strings - the original string is returned unchanged regardless of the deletechars value. The same function works correctly for bytes arrays, indicating this is a str-specific implementation bug rather than an intentional API difference.

## Fix

The implementation should handle the deletechars parameter for Unicode strings similar to how it handles bytes arrays, by filtering out characters present in deletechars from the result.