# Bug Report: numpy.strings.replace String Truncation

**Target**: `numpy.strings.replace`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`numpy.strings.replace()` silently truncates results when the replacement string is longer than the original, due to insufficient output array dtype allocation.

## Property-Based Test

```python
import numpy as np
import numpy.strings as nps
from hypothesis import given, strategies as st

@given(
    st.text(min_size=1, max_size=5),
    st.text(min_size=1, max_size=2),
    st.text(min_size=2, max_size=10)
)
def test_replace_preserves_content(s, old, new):
    arr = np.array([s])
    numpy_result = nps.replace(arr, old, new)[0]
    python_result = s.replace(old, new)
    assert str(numpy_result) == python_result
```

**Failing input**: `s='0', old='0', new='00'`

## Reproducing the Bug

```python
import numpy as np
import numpy.strings as nps

s = '0'
arr = np.array([s])
result = nps.replace(arr, '0', '00')

print(f'Expected: "00"')
print(f'Got: "{result[0]}"')
assert result[0] == '00'
```

## Why This Is A Bug

The function claims to behave like `str.replace()` element-wise, but silently truncates results when the output should be longer than the input. Python's `'0'.replace('0', '00')` correctly returns `'00'`, but numpy.strings.replace returns `'0'` (truncated).

The root cause is that when the input array has dtype `<U1` (max 1 character), the output array is allocated with the same dtype, even though the replacement operation should produce a 2-character string.

## Fix

The replace function should calculate the maximum possible output length and allocate the output array with sufficient dtype size. Similar to how `ljust()` correctly handles this:

```python
def fix_example():
    s = 'a'
    arr = np.array([s])

    result_ljust = nps.ljust(arr, 5)
    print(result_ljust.dtype)

    result_replace = nps.replace(arr, 'a', 'aaaaa')
    print(result_replace.dtype)
```

The replace implementation needs to pre-calculate or dynamically grow the output dtype based on the maximum possible string length after replacement.