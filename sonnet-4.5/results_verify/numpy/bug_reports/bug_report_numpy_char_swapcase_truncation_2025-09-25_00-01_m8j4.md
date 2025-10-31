# Bug Report: numpy.char.swapcase Truncates Multi-Character Case Mappings

**Target**: `numpy.char.swapcase`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`numpy.char.swapcase` truncates the output when Unicode case conversion produces more characters than the input, unlike Python's `str.swapcase()`. This affects characters like German ß and ligatures.

## Property-Based Test

```python
import numpy as np
import numpy.char
from hypothesis import given, strategies as st

@given(st.lists(st.text(), min_size=1))
def test_swapcase_involution(strings):
    arr = np.array(strings)
    result = numpy.char.swapcase(numpy.char.swapcase(arr))
    expected = arr
    assert np.array_equal(result, expected)
```

**Failing input**: `strings=['ß']`

## Reproducing the Bug

```python
import numpy as np
import numpy.char

s = 'ß'
arr = np.array([s])
result = numpy.char.swapcase(arr)

print(f'Input: {s!r}')
print(f'Python str.swapcase(): {s.swapcase()!r}')
print(f'numpy.char.swapcase(): {result[0]!r}')

assert result[0] == s.swapcase()
```

**Output**:
```
Input: 'ß'
Python str.swapcase(): 'SS'
numpy.char.swapcase(): 'S'
AssertionError
```

Another example with ligature:
```python
s = 'ﬃ'
print(f'Python: {s.swapcase()!r}')
print(f'numpy: {numpy.char.swapcase(np.array([s]))[0]!r}')
```

**Output**:
```
Python: 'FFI'
numpy: 'F'
```

## Why This Is A Bug

1. The docstring claims to call `str.swapcase` element-wise, but behavior differs from Python
2. Python's `str.swapcase()` correctly expands 'ß' to 'SS' (following Unicode case mapping rules)
3. `numpy.char.swapcase` truncates 'SS' to 'S' to fit the fixed-width array dtype
4. This causes data corruption for German text and other Unicode strings with special case mappings
5. The truncation happens silently without warning

Impact: Users processing German or other international text will get incorrect results.

## Fix

The root cause is numpy's fixed-width Unicode string dtype (`<U1` for single character). When swapcase expands 'ß' to 'SS', numpy truncates to fit.

Possible fixes:
1. **Resize array**: Detect when case conversion changes length and resize the output array
2. **Raise error**: Explicitly error when truncation would occur
3. **Document limitation**: At minimum, document this behavior

The proper fix would require numpy.char operations to dynamically adjust output array sizes based on actual result lengths, similar to how Python strings work.