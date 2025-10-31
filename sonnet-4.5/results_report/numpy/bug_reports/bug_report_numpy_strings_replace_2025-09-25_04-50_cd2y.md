# Bug Report: numpy.strings.replace Single-Character String Bug

**Target**: `numpy.strings.replace`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `numpy.strings.replace` function fails to replace text when the entire string is a single character being replaced with a longer string.

## Property-Based Test

```python
import numpy as np
import numpy.strings as ns
from hypothesis import given, strategies as st, settings, assume


@given(
    st.lists(st.text(alphabet=st.characters(min_codepoint=97, max_codepoint=122), min_size=0, max_size=20), min_size=1, max_size=20),
    st.text(alphabet=st.characters(min_codepoint=97, max_codepoint=122), min_size=0, max_size=3),
    st.text(alphabet=st.characters(min_codepoint=97, max_codepoint=122), min_size=0, max_size=3)
)
@settings(max_examples=500)
def test_replace_all_occurrences(string_list, old, new):
    assume(old != "")
    arr = np.array(string_list)
    result = ns.replace(arr, old, new)

    for i, s in enumerate(arr):
        expected = s.replace(old, new)
        actual = result[i]
        assert actual == expected
```

**Failing input**: `string_list=['a'], old='a', new='aa'`

## Reproducing the Bug

```python
import numpy as np
import numpy.strings as ns

arr = np.array(['a'])
result = ns.replace(arr, 'a', 'aa')

print("Input:", arr)
print("Result:", result)
print("Expected:", ['aa'])
```

## Why This Is A Bug

In Python, `'a'.replace('a', 'aa')` correctly returns `'aa'`. However, `numpy.strings.replace(['a'], 'a', 'aa')` incorrectly returns `['a']` (unchanged).

The bug occurs specifically when:
1. The string contains only the character(s) being replaced
2. The replacement is longer than the original

This appears to be a buffer allocation or length calculation issue in the underlying implementation. When strings have additional characters (e.g., `'ab'.replace('a', 'aa')` → `'aab'`), the function works correctly.

## Fix

This requires investigation of the C-level implementation in numpy's string operations. The issue likely involves:
- Incorrect pre-allocation of the result buffer size
- Using the original string length instead of calculating the new length after replacement
- A C-string termination issue when the result would be longer than the input