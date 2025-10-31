# Bug Report: numpy.strings Comparison Operators Ignore Trailing Null Characters

**Target**: `numpy.strings.not_equal`, `numpy.strings.less`, `numpy.strings.greater_equal`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

Multiple comparison operators in numpy.strings (`not_equal`, `less`, `greater_equal`) incorrectly treat strings with trailing null characters as equal to their counterparts without those nulls, producing incorrect comparison results.

## Property-Based Test

```python
import numpy as np
import numpy.strings as nps
from hypothesis import given, strategies as st, settings

@given(st.lists(st.text(), min_size=1).map(lambda x: np.array(x, dtype=str)),
       st.lists(st.text(), min_size=1).map(lambda x: np.array(x, dtype=str)))
@settings(max_examples=1000)
def test_comparison_consistency(arr1, arr2):
    if len(arr1) == len(arr2):
        for i in range(len(arr1)):
            np_less = nps.less(arr1[i:i+1], arr2[i:i+1])[0]
            py_less = arr1[i] < arr2[i]
            assert np_less == py_less
```

**Failing input**: `arr1 = np.array(['a'], dtype=str)`, `arr2 = np.array(['a\x00'], dtype=str)`

## Reproducing the Bug

```python
import numpy as np
import numpy.strings as nps

s1, s2 = 'a', 'a\x00'
arr1 = np.array([s1], dtype=str)
arr2 = np.array([s2], dtype=str)

ops = [
    ('not_equal', nps.not_equal, lambda a, b: a != b),
    ('less', nps.less, lambda a, b: a < b),
    ('greater_equal', nps.greater_equal, lambda a, b: a >= b),
]

for name, np_op, py_op in ops:
    np_result = np_op(arr1, arr2)[0]
    py_result = py_op(s1, s2)
    print(f"{name:15}: Python={py_result}, NumPy={np_result}")
```

Output:
```
not_equal      : Python=True, NumPy=False
less           : Python=True, NumPy=False
greater_equal  : Python=False, NumPy=True
```

## Why This Is A Bug

1. **Violates comparison semantics**: Different strings produce identical comparison results because trailing nulls are ignored.

2. **Inconsistent with Python**: Python correctly distinguishes `'a'` from `'a\x00'` in all comparisons.

3. **Breaks ordering properties**: String ordering is corrupted, affecting sorting and comparison-based algorithms.

4. **Related to equal() bug**: This is the same root cause as the `equal()` bug - all comparisons treat trailing nulls as non-existent.

## Fix

This shares the same root cause as the `numpy.strings.equal` bug. The underlying string comparison implementation is treating strings as null-terminated (C-style) rather than using length-aware comparison. All comparison operators need to compare the full string content including trailing null characters.