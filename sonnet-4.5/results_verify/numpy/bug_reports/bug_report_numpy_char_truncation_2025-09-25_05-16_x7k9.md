# Bug Report: numpy.char Functions Truncate Results When Strings Expand

**Target**: `numpy.char.upper()`, `numpy.char.swapcase()`, `numpy.char.replace()`, `numpy.char.translate()`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

Multiple `numpy.char` functions silently truncate results when operations would expand strings beyond their original width. This affects `upper()`, `swapcase()`, `replace()`, and `translate()`, causing data loss for Unicode ligatures (ﬀ, ﬁ, ﬂ, etc.), German sharp S (ß), string replacements, and character translations.

## Property-Based Test

```python
import numpy as np
import numpy.char
from hypothesis import given, strategies as st

@given(st.lists(st.text(min_size=1, max_size=50), min_size=1, max_size=10))
def test_upper_matches_python(strings):
    for s in strings:
        arr = np.array([s])
        numpy_result = numpy.char.upper(arr)[0]
        python_result = s.upper()
        assert numpy_result == python_result
```

**Failing input**: `strings=['ﬀ']` (LATIN SMALL LIGATURE FF, U+FB00)

## Reproducing the Bug

### Bug 1: upper() truncates ligatures

```python
import numpy as np
import numpy.char

arr = np.array(['ﬀ'])
result = numpy.char.upper(arr)[0]
print(f"Expected: 'FF'")
print(f"Got: '{result}'")
```

Output: `Got: 'F'` (should be 'FF')

### Bug 2: swapcase() truncates expansions

```python
arr = np.array(['ß'])
result = numpy.char.swapcase(arr)[0]
print(f"Expected: 'SS'")
print(f"Got: '{result}'")
```

Output: `Got: 'S'` (should be 'SS')

### Bug 3: replace() truncates when replacement expands string

```python
arr = np.array(['0'])
result = numpy.char.replace(arr, '0', '00')[0]
print(f"Expected: '00'")
print(f"Got: '{result}'")
```

Output: `Got: '0'` (should be '00')

### Bug 4: translate() truncates when translation expands characters

```python
translation_table = str.maketrans({'a': 'AA'})
arr = np.array(['a'])
result = numpy.char.translate(arr, translation_table)[0]
print(f"Expected: 'AA'")
print(f"Got: '{result}'")
```

Output: `Got: 'A'` (should be 'AA')

## Why This Is A Bug

The docstrings claim these functions call Python's equivalent methods element-wise:
- `upper()`: "Call str.upper element-wise"
- `swapcase()`: "Calls str.swapcase element-wise"
- `replace()`: "return a copy of the string with all occurrences of substring old replaced by new"
- `translate()`: "Calls str.translate element-wise"

However, they do not produce the same results as Python's methods when operations expand string length.

This violates:
1. **API contract**: Claims to call Python's str methods but produces different results
2. **Data integrity**: Silently loses information during operations
3. **Unicode correctness**: Fails to handle standard Unicode case mappings properly

## Root Cause

NumPy uses fixed-width string arrays. When an operation would expand a string beyond its original width (e.g., 'ß' → 'SS', or '0' → '00'), the result is truncated. The array dtype determines the maximum width (e.g., `<U1` for 1-character strings), and results exceeding this width are silently truncated to fit.

## Fix

The fix requires dynamically calculating the output string width before performing operations. High-level approach:

1. Pre-scan inputs to compute maximum width needed after operation
2. Create output array with sufficient width
3. Perform the operation

Example for `upper()`:
```python
def upper(a):
    max_width = max(len(str(s).upper()) for s in a.flat)
    out = np.empty(a.shape, dtype=f'<U{max_width}')
    # Apply upper() to each element
    return out
```

However, implementing this properly requires changes to numpy's internal C implementation of these string operations.