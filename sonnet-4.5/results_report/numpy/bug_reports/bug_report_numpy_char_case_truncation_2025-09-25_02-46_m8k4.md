# Bug Report: numpy.char Case Functions Truncate Unicode Expansions

**Target**: `numpy.char.upper()`, `numpy.char.capitalize()`, `numpy.char.swapcase()`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`numpy.char.upper()`, `numpy.char.capitalize()`, and `numpy.char.swapcase()` truncate Unicode characters that expand in length during case conversion (e.g., 'ß' → 'SS', 'ﬁ' → 'FI'), causing silent data corruption.

## Property-Based Test

```python
import numpy.char as char
from hypothesis import given, strategies as st, settings

@given(st.text(min_size=0, max_size=100))
@settings(max_examples=500, deadline=None)
def test_upper_matches_python(s):
    numpy_result = char.upper(s)
    numpy_str = str(numpy_result) if hasattr(numpy_result, 'item') else numpy_result
    python_result = s.upper()
    assert numpy_str == python_result
```

**Failing input**: `s='ß'`

## Reproducing the Bug

```python
import numpy.char as char

test_cases = [
    ('ß', 'upper'),
    ('straße', 'upper'),
    ('ﬁ', 'upper'),
    ('ß', 'capitalize'),
]

for s, func_name in test_cases:
    numpy_func = getattr(char, func_name)
    python_func = getattr(str, func_name)

    numpy_result = numpy_func(s).item()
    python_result = python_func(s)

    print(f"{func_name}({repr(s)}):")
    print(f"  numpy:  {repr(numpy_result)} (length {len(numpy_result)})")
    print(f"  python: {repr(python_result)} (length {len(python_result)})")

    if numpy_result != python_result:
        print(f"  TRUNCATED!")
```

Output:
```
upper('ß'):
  numpy:  'S' (length 1)
  python: 'SS' (length 2)
  TRUNCATED!

upper('straße'):
  numpy:  'STRASS' (length 6)
  python: 'STRASSE' (length 7)
  TRUNCATED!

upper('ﬁ'):
  numpy:  'F' (length 1)
  python: 'FI' (length 2)
  TRUNCATED!
```

## Why This Is A Bug

1. **Silent data corruption**: Characters are truncated without any warning or error
2. **Violates documented behavior**: Docstrings state these functions "Call `str.{method}` element-wise", but results differ from Python's str methods
3. **Affects real-world text**: German text with 'ß', ligatures like 'ﬁ', and other Unicode characters are common
4. **Breaks case-insensitive operations**: `upper('ß')` should equal `'SS'` for proper case-insensitive comparison
5. **Violates Unicode standards**: Unicode case mapping is well-defined, and these transformations are specified to expand certain characters

## Fix

The issue stems from numpy's fixed-size character arrays. When case conversion expands string length, output is truncated to fit the original size.

The fix requires:
1. Pre-calculating maximum possible expansion for each character (some characters expand from 1 to 2 or 3 characters)
2. Allocating output arrays with sufficient capacity before case conversion
3. Alternatively, document this limitation clearly and raise an error when expansion would occur

This requires changes in the string ufunc implementations in numpy's core C/Cython code.