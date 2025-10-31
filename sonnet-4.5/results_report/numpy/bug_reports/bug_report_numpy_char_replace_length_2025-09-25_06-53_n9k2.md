# Bug Report: numpy.char.replace Incorrectly Matches Patterns Longer Than String

**Target**: `numpy.char.replace`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`numpy.char.replace()` incorrectly performs replacements when the search pattern is longer than the string being searched, returning truncated replacement text. Python's `str.replace()` correctly returns the original string unchanged when the pattern cannot possibly match.

## Property-Based Test

```python
import numpy.char as char
from hypothesis import given, strategies as st, settings, assume

@given(st.text(min_size=1, max_size=20), st.text(min_size=1, max_size=20))
@settings(max_examples=1000)
def test_replace_respects_pattern_length(s, old):
    assume(len(old) > len(s))

    new = 'REPLACEMENT'

    py_result = s.replace(old, new)
    np_result = str(char.replace(s, old, new))

    if py_result != np_result:
        raise AssertionError(
            f"replace({repr(s)}, {repr(old)}, {repr(new)}): "
            f"Python={repr(py_result)}, NumPy={repr(np_result)}"
        )
```

**Failing input**: `s='0'`, `old='00'`

## Reproducing the Bug

```python
import numpy.char as char

s = '0'
old = '00'
new = 'REPLACEMENT'

py_result = s.replace(old, new)
np_result = str(char.replace(s, old, new))

print(f"Python: '{s}'.replace('{old}', '{new}') = {repr(py_result)}")
print(f"NumPy:  char.replace('{s}', '{old}', '{new}') = {repr(np_result)}")
```

Output:
```
Python: '0'.replace('00', 'REPLACEMENT') = '0'
NumPy:  char.replace('0', '00', 'REPLACEMENT') = 'R'
```

## Why This Is A Bug

1. **Impossible match triggers replacement**: The pattern '00' cannot exist in the string '0', yet numpy performs a replacement anyway.

2. **Truncates replacement**: The function returns only the first character 'R' of 'REPLACEMENT', suggesting severely broken logic.

3. **Data corruption**: Users expect unchanged strings when patterns don't match, but get corrupted data instead.

4. **Violates API contract**: The docstring claims to call `str.replace` element-wise, but produces completely different results.

5. **Consistent with null byte bug**: This appears related to the null byte replacement bugs - the function's pattern matching logic is fundamentally broken.

## Fix

The replace function has multiple critical bugs in its pattern matching and replacement logic:
- It performs replacements when patterns don't match
- It truncates replacement strings
- It mishandles null bytes (documented in separate bug report)

The implementation needs a complete review of its string matching and replacement logic to ensure it:
- Only replaces when the pattern actually exists in the string
- Uses the complete replacement string, not just the first character
- Properly handles edge cases like empty strings, null bytes, and patterns longer than the string