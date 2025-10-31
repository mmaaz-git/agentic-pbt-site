# Bug Report: numpy.strings.mod Null Character Truncation

**Target**: `numpy.strings.mod`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `numpy.strings.mod` function incorrectly truncates strings at null characters (`\x00`) when performing string formatting, treating null as a string terminator instead of a valid Unicode character. This behavior differs from Python's built-in `%` operator.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy as np
import numpy.strings as nps

@given(st.lists(st.just("%s"), min_size=1, max_size=5),
       st.lists(st.text(min_size=0, max_size=10), min_size=1, max_size=5))
def test_mod_string_formatting(format_strings, values):
    """Property: mod with %s should contain the value"""
    assume(len(format_strings) == len(values))

    fmt_arr = np.array(format_strings, dtype='U')
    val_arr = np.array(values, dtype='U')

    result = nps.mod(fmt_arr, val_arr)

    # Each result should contain the corresponding value
    for i in range(len(result)):
        assert values[i] in str(result[i])
```

**Failing input**: `format_strings=['%s'], values=['\x00']`

## Reproducing the Bug

```python
import numpy as np
import numpy.strings as nps

# Demonstrate the bug
fmt_arr = np.array(['%s'], dtype='U')
val_arr = np.array(['\x00'], dtype='U')
result = nps.mod(fmt_arr, val_arr)

print(f"Result: {repr(result)}")
print(f"Result length: {nps.str_len(result)[0]}")

# Compare with Python's behavior
python_result = '%s' % '\x00'
print(f"Python result: {repr(python_result)}")
print(f"Python result length: {len(python_result)}")

# Bug demonstration
assert len(python_result) == 1  # Python correctly handles null
assert nps.str_len(result)[0] == 0  # NumPy truncates at null (BUG)
assert result[0] == ''  # Result is empty string instead of '\x00'
```

Output:
```
Result: array([''], dtype='<U1')
Result length: 0
Python result: '\x00'
Python result length: 1
```

## Why This Is A Bug

The null character (`\x00`) is a valid Unicode character and should be treated as such in Unicode string operations. Python's `%` operator correctly preserves null characters in string formatting: `'%s' % '\x00'` returns `'\x00'` (a string of length 1).

NumPy's `strings.mod` function incorrectly treats `\x00` as a C-style string terminator, truncating the result to an empty string. This is inconsistent with:
1. Python's string formatting behavior
2. NumPy's handling of other control characters (e.g., `\x01`, `\t`, `\n` all work correctly)
3. The expected behavior of Unicode strings, which should not be null-terminated

This breaks the reasonable expectation that `np.strings.mod` behaves like element-wise application of Python's `%` operator.

## Fix

The issue likely stems from using C-style string functions that treat null as a terminator in the implementation of `mod`. The fix would require ensuring that string length is determined explicitly rather than relying on null-termination.

Based on the source code at `numpy/_core/strings.py` lines 235-268, the `mod` function uses `_vec_string` which likely delegates to lower-level C code. The fix would need to be in the C implementation layer to properly handle null characters in Unicode strings by using explicit length tracking instead of null-termination.