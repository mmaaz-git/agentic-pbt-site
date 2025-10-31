# Bug Report: numpy.char Case Transformation Functions Truncate Multi-Character Results

**Target**: `numpy.char.upper`, `numpy.char.swapcase`, `numpy.char.capitalize`, `numpy.char.title`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

NumPy's case transformation functions silently truncate results when Unicode case transformations produce multi-character outputs, violating their documented contracts to call Python's `str.upper()`, `str.swapcase()`, etc. element-wise.

## Property-Based Test

```python
import numpy as np
import numpy.char
from hypothesis import given, strategies as st


@given(st.lists(st.text(), min_size=1))
def test_swapcase_matches_python(strings):
    arr = np.array(strings)
    numpy_result = numpy.char.swapcase(arr)

    for i in range(len(strings)):
        python_result = strings[i].swapcase()
        assert numpy_result[i] == python_result
```

**Failing input**: `strings=['ẖ']`

## Reproducing the Bug

```python
import numpy as np
import numpy.char

test_char = 'ẖ'
arr = np.array([test_char])

numpy_result = numpy.char.swapcase(arr)[0]
python_result = test_char.swapcase()

print(f"Input: '{test_char}' (U+1E96, LATIN SMALL LETTER H WITH LINE BELOW)")
print(f"Python str.swapcase: '{python_result}' = 'H' + COMBINING MACRON BELOW (len={len(python_result)})")
print(f"NumPy char.swapcase: '{numpy_result}' = 'H' only (len={len(numpy_result)})")
print(f"Match: {str(numpy_result) == python_result}")

print("\nOther affected characters:")
for char in ['ẖ', 'ǰ', 'ß']:
    arr = np.array([char])
    print(f"  '{char}' → NumPy: {repr(str(numpy.char.upper(arr)[0]))}, Python: {repr(char.upper())}")
```

Output:
```
Input: 'ẖ' (U+1E96, LATIN SMALL LETTER H WITH LINE BELOW)
Python str.swapcase: 'H̱' = 'H' + COMBINING MACRON BELOW (len=2)
NumPy char.swapcase: 'H' = 'H' only (len=1)
Match: False

Other affected characters:
  'ẖ' → NumPy: 'H', Python: 'H̱'
  'ǰ' → NumPy: 'J', Python: 'J̌'
  'ß' → NumPy: 'S', Python: 'SS'
```

## Why This Is A Bug

1. **Violates documented API contract**: The docstring explicitly states these functions call `str.upper()`, `str.swapcase()`, etc. element-wise, but they don't match Python's behavior.

2. **Silent data corruption**: The functions silently truncate/lose information without warning or error, making round-trip operations impossible.

3. **Breaks fundamental properties**: For characters where case transformation produces multi-character results (German ß → SS, characters with combining diacritics), NumPy produces incorrect results.

4. **Affects multiple functions**: `upper()`, `swapcase()`, `capitalize()`, and `title()` all exhibit this bug.

## Fix

The issue appears to be that NumPy's string dtype has fixed-width character storage, preventing expansion when case transformations require more characters. A proper fix would need to:

1. Detect when case transformations expand character count
2. Allocate larger arrays to accommodate expanded results
3. Update the string dtype width accordingly

This is a non-trivial fix requiring changes to NumPy's core string handling. A minimal patch would at minimum document this limitation and potentially warn users when truncation occurs.