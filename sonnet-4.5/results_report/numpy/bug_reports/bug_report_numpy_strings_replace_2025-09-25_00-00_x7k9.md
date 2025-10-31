# Bug Report: numpy.strings.replace Silent Truncation

**Target**: `numpy.strings.replace`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`numpy.strings.replace` silently truncates replacement results when the input array's dtype is too small to hold the replaced string, causing data corruption.

## Property-Based Test

```python
import numpy as np
import numpy.strings as nps
from hypothesis import given, strategies as st, assume, settings


@settings(max_examples=1000)
@given(st.lists(st.text(min_size=1, max_size=50), min_size=1, max_size=20), st.text(min_size=0, max_size=5), st.text(min_size=0, max_size=5))
def test_replace_matches_python(strings, old, new):
    assume(len(old) > 0)
    arr = np.array(strings)
    replaced = nps.replace(arr, old, new)

    for i, s in enumerate(strings):
        expected = s.replace(old, new)
        assert replaced[i] == expected
```

**Failing input**: `strings=['0'], old='0', new='00'`

## Reproducing the Bug

```python
import numpy as np
import numpy.strings as nps

arr = np.array(['0'])
result = nps.replace(arr, '0', '00')

print(f'Input: {arr}, dtype: {arr.dtype}')
print(f'Result: {result}, dtype: {result.dtype}')
print(f'Expected: ["00"]')
print(f'Actual: {result}')

assert result[0] == '00', f'Expected "00", got {result[0]}'
```

Output:
```
Input: ['0'], dtype: <U1
Result: ['0'], dtype: <U1
Expected: ["00"]
Actual: ['0']
AssertionError: Expected "00", got 0
```

## Why This Is A Bug

1. **API Contract Violation**: The docstring states that `replace` behaves like `str.replace` element-wise, but `'0'.replace('0', '00')` returns `'00'` while `numpy.strings.replace(np.array(['0']), '0', '00')` returns `'0'`.

2. **Silent Data Corruption**: The function silently truncates the result to fit the input dtype without raising an error or warning, leading to incorrect results.

3. **Inconsistent Behavior**: Other numpy.strings functions like `center` correctly expand the dtype when needed (e.g., `np.array(['a'], dtype='<U1')` correctly becomes dtype `<U5` after `center(5)`), but `replace` fails to do so in certain cases.

## Fix

The fix should ensure that `replace` computes the necessary output dtype size and creates an appropriately-sized output array before performing the replacement. The function should either:

1. Pre-compute the maximum possible output length based on the inputs and allocate the output array accordingly, OR
2. Dynamically resize the output dtype after computing the replacement

Similar to how `center` handles dtype expansion:
```python
# center correctly expands dtype
arr = np.array(['a'], dtype='<U1')
result = nps.center(arr, 5, 'x')  # Creates dtype <U5
```

The `replace` function should follow the same pattern when the replacement would exceed the input dtype size.