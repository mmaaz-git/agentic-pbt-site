# Bug Report: numpy.char Case Conversion Truncates Multi-Character Results

**Target**: `numpy.char.upper`, `numpy.char.lower`, `numpy.char.swapcase`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

NumPy's `char.upper()`, `char.lower()`, and `char.swapcase()` truncate results to a single character when Unicode case conversion produces multiple characters (e.g., 'ß'.upper() should return 'SS' but returns 'S').

## Property-Based Test

```python
@given(st.text())
def test_swapcase_involution(text):
    arr = np.array([text])
    result = numpy.char.swapcase(numpy.char.swapcase(arr))
    assert result[0] == text
```

**Failing input**: `'ß'`

```python
@given(st.text())
def test_upper_lower_idempotence(text):
    arr = np.array([text])
    result1 = numpy.char.lower(numpy.char.upper(arr))
    result2 = numpy.char.lower(arr)
    assert result1[0] == result2[0]
```

**Failing input**: `'ß'`

## Reproducing the Bug

```python
import numpy as np
import numpy.char

arr = np.array(['ß'])
result = numpy.char.upper(arr)
print(f"Expected: 'SS', Got: {repr(result[0])}")

swapped_twice = numpy.char.swapcase(numpy.char.swapcase(arr))
print(f"swapcase(swapcase('ß')) = {repr(swapped_twice[0])} (expected 'ß')")

upper_lower = numpy.char.lower(numpy.char.upper(arr))
print(f"lower(upper('ß')) = {repr(upper_lower[0])} (expected 'ß')")
```

Output:
```
Expected: 'SS', Got: np.str_('S')
swapcase(swapcase('ß')) = np.str_('s') (expected 'ß')
lower(upper('ß')) = np.str_('s') (expected 'ß')
```

## Why This Is A Bug

The documentation for `numpy.char.upper()` states: "Calls `str.upper` element-wise." However, Python's `str.upper()` returns `'SS'` for `'ß'`, while NumPy returns only `'S'`.

This violates fundamental mathematical properties:
- **Involution**: `swapcase(swapcase(x))` should equal `x`
- **Idempotence**: `lower(upper(x))` should equal `lower(x)`

The bug affects all Unicode characters whose case conversion produces multiple characters:
- `'ß'` (German sharp s): should become `'SS'`
- `'ﬃ'` (ligature ffi): should become `'FFI'`
- `'ﬁ'` (ligature fi): should become `'FI'`
- `'ﬆ'` (ligature st): should become `'ST'`

This is a data corruption bug that silently produces incorrect results.

## Fix

This is likely caused by NumPy allocating fixed-size character arrays that cannot accommodate the length change during case conversion. A proper fix would require:

1. Pre-calculating the maximum length after case conversion
2. Allocating appropriately sized output arrays
3. Handling variable-length results correctly

This is a fundamental limitation in NumPy's fixed-width string handling and may require significant architectural changes to fix properly.