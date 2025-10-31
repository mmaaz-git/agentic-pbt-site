# Bug Report: numpy.strings.equal Ignores Trailing Null Characters

**Target**: `numpy.strings.equal`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`numpy.strings.equal()` incorrectly treats strings with trailing null characters as equal to their counterparts without those trailing nulls. For example, `'a'` and `'a\x00'` are considered equal when they should not be.

## Property-Based Test

```python
import numpy as np
import numpy.strings as nps
from hypothesis import given, strategies as st, settings

@given(st.lists(st.text(), min_size=1).map(lambda x: np.array(x, dtype=str)),
       st.lists(st.text(), min_size=1).map(lambda x: np.array(x, dtype=str)))
@settings(max_examples=1000)
def test_equal_consistency(arr1, arr2):
    if len(arr1) == len(arr2):
        np_eq = nps.equal(arr1, arr2)
        for i in range(len(arr1)):
            py_eq = arr1[i] == arr2[i]
            assert np_eq[i] == py_eq
```

**Failing input**: `arr1 = np.array(['a'], dtype=str)`, `arr2 = np.array(['a\x00'], dtype=str)`

## Reproducing the Bug

```python
import numpy as np
import numpy.strings as nps

test_cases = [
    ('a', 'a\x00'),
    ('abc', 'abc\x00'),
    ('', '\x00'),
]

for s1, s2 in test_cases:
    arr1 = np.array([s1], dtype=str)
    arr2 = np.array([s2], dtype=str)
    np_eq = nps.equal(arr1, arr2)[0]
    py_eq = s1 == s2
    print(f"equal({repr(s1):8}, {repr(s2):10}): Python={py_eq}, NumPy={np_eq}")
```

Output:
```
equal('a'    , 'a\x00'   ): Python=False, NumPy=True
equal('abc'  , 'abc\x00' ): Python=False, NumPy=True
equal(''     , '\x00'    ): Python=False, NumPy=True
```

## Why This Is A Bug

1. **Violates equality semantics**: Two different strings are considered equal, breaking fundamental equality properties.

2. **Inconsistent with Python**: Python correctly treats these as different strings.

3. **Breaks transitivity and other properties**: If `'a' == 'a\x00'` and `'a\x00' != 'a\x00\x00'`, then we could have `'a' == 'a\x00'` but `'a' != 'a\x00\x00'`, which seems inconsistent.

4. **Data integrity**: Code relying on equality checks for validation or deduplication will incorrectly merge distinct strings.

## Fix

This is likely related to the slice bug - the comparison logic is treating trailing null characters as insignificant, possibly by using C-style null-terminated string comparison instead of length-aware comparison. The implementation should compare strings including all characters up to their actual length, not stopping at the first null character.