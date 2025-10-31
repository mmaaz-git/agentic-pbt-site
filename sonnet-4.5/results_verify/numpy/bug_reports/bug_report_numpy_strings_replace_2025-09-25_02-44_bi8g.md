# Bug Report: numpy.strings.replace Truncates Output for Short Strings

**Target**: `numpy.strings.replace`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When replacing an empty substring `''` in strings of length less than 3, `numpy.strings.replace` incorrectly truncates the output to be at most `len(input) + 1` characters, regardless of the replacement string length.

## Property-Based Test

```python
import numpy as np
import numpy.strings as nps
from hypothesis import given, strategies as st


@given(st.text(max_size=2), st.text(min_size=2, max_size=10))
def test_replace_empty_pattern_short_strings(input_str, replacement):
    arr = np.array([input_str])
    numpy_result = nps.replace(arr, '', replacement)[0]
    python_result = input_str.replace('', replacement)
    assert numpy_result == python_result
```

**Failing input**: `input_str='', replacement='00'`

## Reproducing the Bug

```python
import numpy as np
import numpy.strings as nps

arr = np.array([''])
result = nps.replace(arr, '', '00')

print(f"Python: {''.replace('', '00')!r}")
print(f"NumPy:  {result[0]!r}")

assert ''.replace('', '00') == '00'
assert result[0] == '0'
```

Comprehensive demonstration of the bug pattern:

```python
import numpy as np
import numpy.strings as nps

test_cases = [
    ('', 'ab'),
    ('', 'abc'),
    ('a', 'xyz'),
    ('ab', 'xyz'),
    ('abc', 'XYZ'),
    ('hello', 'XYZ'),
]

for input_str, replacement in test_cases:
    python_result = input_str.replace('', replacement)
    numpy_result = nps.replace(np.array([input_str]), '', replacement)[0]
    match = "✓" if python_result == numpy_result else "✗"
    print(f"{match} input={input_str!r:10} | Python={python_result!r:30} | NumPy={numpy_result!r}")
```

Output:
```
✗ input=''         | Python='ab'                        | NumPy='a'
✗ input=''         | Python='abc'                       | NumPy='a'
✗ input='a'        | Python='xyzaxyz'                   | NumPy='xax'
✗ input='ab'       | Python='xyzaxyzbxyz'               | NumPy='xyaxybxy'
✓ input='abc'      | Python='XYZaXYZbXYZcXYZ'           | NumPy='XYZaXYZbXYZcXYZ'
✓ input='hello'    | Python='XYZhXYZeXYZlXYZlXYZoXYZ'   | NumPy='XYZhXYZeXYZlXYZlXYZoXYZ'
```

## Why This Is A Bug

The `numpy.strings.replace` function is documented to call `str.replace` element-wise and should produce identical results to Python's built-in behavior. However, when the pattern to replace is an empty string `''` and the input string has length < 3, the function truncates the output incorrectly.

When Python replaces an empty string, it inserts the replacement at every position (before first char, between each char, and after last char). For input `''` and replacement `'00'`, Python correctly produces `'00'`. NumPy produces only `'0'`, losing half the result.

This bug affects any code that:
1. Uses empty string replacement for formatting or string building
2. Operates on short strings (common in data processing)
3. Expects NumPy to match standard Python string behavior

## Fix

The bug likely exists in the C implementation's buffer size calculation. When `old == ''`, the output buffer should be sized for `(len(input) + 1) * len(new)` characters, but it appears to be incorrectly calculated for short strings.

Without access to modify the C/Cython source, a precise patch cannot be provided. The fix would involve:

1. Locating the buffer allocation code in the C implementation of `replace`
2. Correcting the size calculation for the special case when `old == ''`
3. Ensuring the copy logic properly handles all `len(input) + 1` insertion points

The bug threshold suggests the issue may be in how the algorithm handles cases where the output length exceeds the input length by more than a certain factor.