# Bug Report: numpy.char.rpartition Crashes on Null Byte Separator

**Target**: `numpy.char.rpartition`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`numpy.char.rpartition()` raises `ValueError: empty separator` when given `\x00` as separator, while Python's `str.rpartition()` handles it correctly.

## Property-Based Test

```python
import numpy as np
import numpy.char
from hypothesis import given, strategies as st


@given(st.lists(st.text(), min_size=1), st.text())
def test_rpartition_matches_python(strings, sep):
    assume(len(sep) > 0)
    arr = np.array(strings)
    numpy_result = numpy.char.rpartition(arr, sep)

    for i in range(len(strings)):
        python_result = strings[i].rpartition(sep)
        assert tuple(numpy_result[i]) == python_result
```

**Failing input**: `strings=[''], sep='\x00'`

## Reproducing the Bug

```python
import numpy as np
import numpy.char

test_string = ''
sep = '\x00'

arr = np.array([test_string])

python_result = test_string.rpartition(sep)
print(f"Python rpartition: {python_result}")

try:
    numpy_result = numpy.char.rpartition(arr, sep)
    print(f"NumPy rpartition: {tuple(numpy_result[0])}")
except ValueError as e:
    print(f"NumPy rpartition: ValueError: {e}")
```

Output:
```
Python rpartition: ('', '', '')
NumPy rpartition: ValueError: empty separator
```

## Why This Is A Bug

1. **API inconsistency**: NumPy claims to call `str.rpartition` element-wise but rejects valid separators
2. **Incorrect error**: `\x00` is a valid single-character separator, not an empty string
3. **Breaks user code**: Code that works with Python strings fails with NumPy arrays

## Fix

```diff
--- a/numpy/_core/strings.py
+++ b/numpy/_core/strings.py
@@ -1684,7 +1684,7 @@ def rpartition(a, sep, pos=slice(None), out=None):
     if not issubclass(sep_dt.type, (bytes_, str_)):
         raise TypeError(
             f"`sep` must be a string or bytes array, not {sep_dt}")
-    if _is_empty_array(sep):
+    if _is_empty_array(sep) or (np.isscalar(sep) and len(sep) == 0):
         raise ValueError("empty separator")
```

Note: The actual fix needs to distinguish between empty string `""` (invalid) and null byte `"\x00"` (valid). The condition should check `len(sep) == 0` properly accounting for null bytes.