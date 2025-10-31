# Bug Report: numpy.char Case Conversion Silent Truncation

**Target**: `numpy.char.upper()`, `numpy.char.lower()`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`numpy.char.upper()` and `numpy.char.lower()` silently truncate results when case conversion expands character count, violating the documented contract that they call `str.upper()` and `str.lower()` element-wise.

## Property-Based Test

```python
import numpy as np
import numpy.char as char
from hypothesis import given, strategies as st


@given(st.lists(st.text(min_size=1), min_size=1))
def test_consistency_with_python_upper(strings):
    arr = np.array(strings, dtype=str)
    numpy_result = char.upper(arr)
    python_result = np.array([s.upper() for s in strings], dtype=str)
    assert np.array_equal(numpy_result, python_result)
```

**Failing input**: `['ß']`

## Reproducing the Bug

```python
import numpy as np
import numpy.char as char

arr = np.array(['ß'])
result = char.upper(arr)

print(f"Expected: {'ß'.upper()!r}")
print(f"Actual: {result[0]!r}")
```

Output:
```
Expected: 'SS'
Actual: 'S'
```

The same bug affects `lower()`:

```python
arr2 = np.array(['İ'])
result2 = char.lower(arr2)

print(f"Expected: {'İ'.lower()!r}")
print(f"Actual: {result2[0]!r}")
```

Output:
```
Expected: 'i̇'
Actual: 'i'
```

## Why This Is A Bug

The docstrings for `numpy.char.upper()` and `numpy.char.lower()` explicitly state they call `str.upper()` and `str.lower()` element-wise. However, when the input array has auto-inferred dtype (e.g., `U1` for single-character strings) and case conversion expands the character count, the functions silently truncate the result instead of expanding the dtype or raising an error.

This causes **silent data corruption** for legitimate inputs like German text containing 'ß' or Turkish text containing 'İ'. Users expect `char.upper(arr)` to behave like `[s.upper() for s in arr]`, but instead get corrupted data.

## Fix

The fix should ensure case conversion functions automatically expand the output dtype to accommodate the result. One approach:

```diff
--- a/numpy/char/_methods.py
+++ b/numpy/char/_methods.py
@@ -upper_function
-    return _vec_string(a, a.dtype, 'upper')
+    # Pre-compute maximum output length for each element
+    max_len = max(len(str(x).upper()) for x in a.flat)
+    out_dtype = f'U{max_len}'
+    return _vec_string(a, np.dtype(out_dtype), 'upper')
```

Alternatively, raise an error when truncation would occur rather than silently corrupting data.