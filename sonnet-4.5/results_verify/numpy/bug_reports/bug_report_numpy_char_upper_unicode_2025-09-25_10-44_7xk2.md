# Bug Report: numpy.char.upper/lower Unicode Case Transformation

**Target**: `numpy.char.upper`, `numpy.char.lower`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`numpy.char.upper` and `numpy.char.lower` incorrectly handle Unicode case transformations that change the number of characters, such as Greek characters with iota subscripts, causing data loss.

## Property-Based Test

```python
import numpy.char as char
from hypothesis import given, strategies as st, settings


safe_text = st.text(
    alphabet=st.characters(
        blacklist_categories=('Cs',),
        blacklist_characters=' \t\n\r\x00\x0b\x0c'
    ),
    min_size=1
)


@given(safe_text)
@settings(max_examples=1000)
def test_upper_lower_roundtrip(s):
    arr = char.array([s])
    result = char.lower(char.upper(arr))
    expected = s.upper().lower()
    assert result[0] == expected
```

**Failing input**: `'ῂ'` (Greek eta with combining characters)

## Reproducing the Bug

```python
import numpy.char as char

s = 'ῂ'
arr = char.array([s])
upper = char.upper(arr)
lower_upper = char.lower(upper)

python_upper = s.upper()
python_lower_upper = python_upper.lower()

print(f"Input: {repr(s)}")
print(f"Python:  {repr(s)} -> upper: {repr(python_upper)} -> lower: {repr(python_lower_upper)}")
print(f"NumPy:   {repr(s)} -> upper: {repr(upper[0])} -> lower: {repr(lower_upper[0])}")
print(f"NumPy result: {repr(lower_upper[0])}")
print(f"Expected:     {repr(python_lower_upper)}")
```

Output:
```
Input: 'ῂ'
Python:  'ῂ' -> upper: 'ῊΙ' -> lower: 'ὴι'
NumPy:   'ῂ' -> upper: 'Ὴ' -> lower: 'ὴ'
NumPy result: 'ὴ'
Expected:     'ὴι'
```

## Why This Is A Bug

Unicode case transformations can change the number of characters in a string. For example, some Greek characters with iota subscripts expand to two characters when uppercased. Python's `str.upper()` and `str.lower()` handle this correctly, but NumPy's implementation loses characters during transformation.

This violates:
1. The documented behavior that `numpy.char` functions call `str.upper`/`str.lower` "element-wise"
2. Unicode case-folding standards (Unicode Technical Report #21)
3. User expectations that case transformations are reversible operations

The bug causes **silent data loss** when processing text with combining characters or special Unicode case mappings.

## Fix

The fix requires updating NumPy's case transformation implementation to properly handle Unicode normalization and case mappings that change character counts. This likely requires:
1. Pre-allocating sufficient buffer space for character expansion
2. Using proper Unicode case-folding algorithms instead of simple character-by-character transformation
3. Possibly using ICU or Python's built-in Unicode handling