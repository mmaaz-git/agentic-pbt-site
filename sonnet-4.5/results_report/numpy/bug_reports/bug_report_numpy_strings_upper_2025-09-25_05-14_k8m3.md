# Bug Report: numpy.strings.upper Case Conversion Truncation

**Target**: `numpy.strings.upper` (also affects `capitalize`, `swapcase`, `title`)
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

Case conversion functions (`upper`, `capitalize`, `swapcase`, `title`) silently truncate results when Unicode characters expand during conversion (e.g., 'ß' → 'SS'), due to insufficient output array dtype allocation.

## Property-Based Test

```python
import numpy as np
import numpy.strings as nps
from hypothesis import given, strategies as st

@given(st.text())
def test_upper_consistency_with_str(s):
    arr = np.array([s])
    numpy_result = nps.upper(arr)[0]
    python_result = s.upper()
    assert numpy_result == python_result
```

**Failing input**: `s='ß'`

## Reproducing the Bug

```python
import numpy as np
import numpy.strings as nps

s = 'ß'
arr = np.array([s])
result = nps.upper(arr)

print(f'Python result: {repr(s.upper())}')
print(f'NumPy result: {repr(result[0])}')
assert result[0] == s.upper()
```

Output:
```
Python result: 'SS'
NumPy result: 'S'
AssertionError
```

The same bug affects other case conversion functions:
- `capitalize('ß')` returns 'S' instead of 'Ss'
- `swapcase('ß')` returns 'S' instead of 'SS'
- `title('ß')` returns 'S' instead of 'Ss'

And other expanding Unicode characters:
- Ligatures: 'ﬁ'.upper() → 'FI', 'ﬀ'.upper() → 'FF', 'ﬂ'.upper() → 'FL', 'ﬃ'.upper() → 'FFI'

## Why This Is A Bug

These functions claim to call Python's `str.upper()`, `str.capitalize()`, etc. element-wise, but they produce incorrect results by truncating expanded characters. This is silent data corruption - users get wrong results without any error or warning.

The root cause is that when input arrays have small dtypes (e.g., `<U1` for a single character), the output arrays are allocated with the same dtype, causing truncation when Unicode case conversion expands characters.

## Fix

Case conversion functions need to account for potential string expansion. One approach:

1. Pre-scan for expanding characters and calculate max output length
2. Allocate output array with sufficient dtype size
3. Or use a dynamic allocation strategy similar to what `ljust()` does (which correctly handles output size)

The functions should either:
- Calculate the maximum possible expansion (worst case: some characters expand 3x)
- Or perform a two-pass approach: first pass determines max length, second pass does the conversion