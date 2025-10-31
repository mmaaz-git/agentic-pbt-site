# Bug Report: numpy.strings swapcase/upper Truncation with German Eszett

**Target**: `numpy.strings.swapcase`, `numpy.strings.upper`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`numpy.strings.swapcase()` and `numpy.strings.upper()` silently truncate strings when Unicode case conversion increases string length, specifically with German eszett (ß -> SS).

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy as np
import numpy.strings as nps

@given(st.lists(st.text(), min_size=1, max_size=10))
def test_swapcase_preserves_length(strings):
    arr = np.array(strings)
    swapped = nps.swapcase(arr)
    assert all(len(s) == len(sw) for s, sw in zip(arr, swapped))
```

**Failing input**: `['ß', '00']`

## Reproducing the Bug

```python
import numpy as np
import numpy.strings as nps

arr = np.array(['ß'])
result = nps.swapcase(arr)
print(f"Input: 'ß'")
print(f"Expected (Python str.swapcase()): 'SS'")
print(f"Actual (NumPy): '{result[0]}'")

arr2 = np.array(['straße'])
result2 = nps.upper(arr2)
print(f"\nInput: 'straße'")
print(f"Expected (Python str.upper()): 'STRASSE'")
print(f"Actual (NumPy): '{result2[0]}'")
```

## Why This Is A Bug

1. Documentation claims these functions call `str.swapcase()` and `str.upper()` element-wise
2. German eszett 'ß' correctly uppercases to 'SS' in Python (Unicode standard)
3. NumPy truncates 'SS' to 'S', losing the second character
4. Causes silent data corruption - 'straße' becomes 'STRASS' instead of 'STRASSE'
5. Violates the fundamental property that case transformations should preserve semantic content

## Fix

The bug appears to be in the buffer allocation logic. NumPy pre-allocates output buffers based on input string length but doesn't account for case transformations that can increase length. The fix requires:

1. Either dynamically resize output buffers when case conversion increases length
2. Or pre-calculate maximum possible output length (e.g., 2x input for ß -> SS conversions)

This likely affects the C implementation layer where the buffers are allocated.