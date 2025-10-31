# Bug Report: numpy.char.upper() Truncates Unicode Case Expansions

**Target**: `numpy.char.upper()`, `numpy.char.swapcase()`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`numpy.char.upper()` and `numpy.char.swapcase()` truncate Unicode characters that expand in length during case conversion (e.g., German sharp s 'ß' → 'SS'), silently corrupting data.

## Property-Based Test

```python
import numpy.char as char
from hypothesis import given, strategies as st, settings

@given(st.text(min_size=0, max_size=100))
@settings(max_examples=500, deadline=None)
def test_swapcase_involution(s):
    swap1 = char.swapcase(s)
    swap2 = char.swapcase(swap1)
    swap2_str = str(swap2) if hasattr(swap2, 'item') else swap2
    assert swap2_str == s
```

**Failing input**: `s='ß'`

## Reproducing the Bug

```python
import numpy.char as char

s = 'ß'

numpy_upper = char.upper(s).item()
python_upper = s.upper()

print(f"Input: {repr(s)}")
print(f"numpy.char.upper():  {repr(numpy_upper)} (length {len(numpy_upper)})")
print(f"Python str.upper():  {repr(python_upper)} (length {len(python_upper)})")

assert numpy_upper == python_upper
```

Expected: `'SS'` (length 2)
Actual: `'S'` (length 1) - second character truncated

## Why This Is A Bug

1. **Silent data corruption**: The uppercase form is truncated without warning, losing the second 'S'
2. **Violates documented behavior**: The docstring states "Calls `str.upper` element-wise", but the result doesn't match `str.upper()`
3. **Affects real-world text**: German text using 'ß' is common and legitimate
4. **Breaks case-insensitive comparisons**: `upper('ß')` should equal `upper('SS')`, but due to truncation they differ

## Fix

The issue is that numpy uses fixed-size character arrays. When case conversion would expand the string length (as with ß → SS), the result is truncated to fit the original size. The fix requires:

1. Calculating the maximum possible expanded length before case conversion
2. Allocating sufficient space in the output array
3. Or, documenting this limitation clearly and raising an error when truncation would occur

This requires changes in the underlying string ufunc implementation that handles case transformations.