# Bug Report: numpy.char Case Transformation Functions Silently Truncate Unicode

**Target**: `numpy.char.upper()`, `numpy.char.lower()`, and related case transformation functions
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

Case transformation functions (`upper()`, `lower()`, `title()`, `capitalize()`) silently truncate results when Unicode case mapping produces strings longer than the input array's dtype can accommodate, causing data corruption for characters like ß (German eszett), İ (Turkish I with dot), and ligatures (ﬁ, ﬂ, etc.).

## Property-Based Test

```python
import numpy as np
import numpy.char as nc
from hypothesis import given, settings, strategies as st

@given(st.lists(st.text(), min_size=1))
@settings(max_examples=1000)
def test_lower_upper_roundtrip(strings):
    arr = np.array(strings, dtype=str)
    lowered = nc.lower(arr)
    result = nc.lower(nc.upper(arr))
    assert np.array_equal(lowered, result)
```

**Failing input**: `['ß']` (inferred dtype is `<U1`, which cannot hold `'SS'`)

## Reproducing the Bug

```python
import numpy as np
import numpy.char as nc

arr = np.array(['ß'])
print(f"Input dtype: {arr.dtype}")
print(f"Python 'ß'.upper(): 'SS'")

result = nc.upper(arr)
print(f"NumPy result: '{result[0]}'")

arr_adequate = np.array(['ß'], dtype='<U10')
result_adequate = nc.upper(arr_adequate)
print(f"With adequate dtype: '{result_adequate[0]}'")

arr_lower = np.array(['İ'])
result_lower = nc.lower(arr_lower)
print(f"Turkish İ with dtype <U1: '{result_lower[0]}' (expected 'i̇')")
```

Output:
```
Input dtype: <U1
Python 'ß'.upper(): 'SS'
NumPy result: 'S'
With adequate dtype: 'SS'
Turkish İ with dtype <U1: 'i' (expected 'i̇')
```

## Why This Is A Bug

The documentation states these functions "call str.upper/str.lower element-wise", but they deviate from Python's behavior:

- `'ß'.upper()` returns `'SS'` in Python, but `'S'` in NumPy with dtype `<U1`
- `'İ'.lower()` returns `'i̇'` in Python (2 chars), but `'i'` in NumPy with dtype `<U1`
- `'ﬁ'.upper()` returns `'FI'` in Python, but `'F'` in NumPy with dtype `<U1`

This violates documented behavior and causes silent data corruption because:
1. No warning or error is raised when truncation occurs
2. Default dtype inference uses minimum size (e.g., `np.array(['ß'])` → dtype `<U1`)
3. Real-world text processing is affected (German, Turkish, ligatures)

## Fix

The root cause is that case transformation functions preserve the input dtype without checking if the output requires more space. A proper fix would require changes to the NumPy C implementation.

High-level fix approach:
1. Pre-compute the maximum output length needed for each element
2. Create output array with sufficient dtype (e.g., `<U{max_len}`)
3. Or at minimum, raise a warning when truncation would occur

Since this requires C-level changes in NumPy core, the most practical immediate fix is to:
- Document this limitation prominently in the docstrings
- Add a warning when output dtype might be insufficient
- Recommend users explicitly specify adequate dtype for Unicode text