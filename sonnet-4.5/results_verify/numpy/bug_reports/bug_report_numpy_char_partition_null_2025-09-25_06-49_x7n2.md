# Bug Report: numpy.char.partition/rpartition Incorrectly Rejects Null Byte Separator

**Target**: `numpy.char.partition`, `numpy.char.rpartition`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`numpy.char.partition()` and `numpy.char.rpartition()` incorrectly treat null bytes (`\x00`) as empty separators and raise `ValueError`, while Python's `str.partition()` and `str.rpartition()` accept null bytes as valid separators. This violates the documented contract that these functions call the corresponding str methods element-wise.

## Property-Based Test

```python
import numpy.char as char
from hypothesis import given, strategies as st, settings

@given(st.text(min_size=0, max_size=20))
@settings(max_examples=1000)
def test_partition_matches_python_behavior(s):
    sep = '\x00'

    try:
        py_result = s.partition(sep)
    except ValueError:
        py_result = None

    try:
        np_result = char.partition(s, sep)
        if hasattr(np_result, 'shape') and np_result.shape == ():
            np_result = np_result.item()
        np_result = tuple(str(p) for p in np_result)
    except ValueError:
        np_result = None

    if (py_result is None) != (np_result is None):
        raise AssertionError(
            f"partition('{s}', {repr(sep)}) behavior differs: "
            f"Python: {py_result}, NumPy: {np_result}"
        )
```

**Failing input**: Any string with separator `'\x00'`

## Reproducing the Bug

```python
import numpy.char as char

s = 'test'
sep = '\x00'

print("Python str.partition:")
print(s.partition(sep))

print("\nnumpy.char.partition:")
try:
    result = char.partition(s, sep)
    print(result)
except ValueError as e:
    print(f"ValueError: {e}")
```

Output:
```
Python str.partition:
('test', '', '')

numpy.char.partition:
ValueError: empty separator
```

## Why This Is A Bug

1. **Violates documented contract**: The docstring states "Calls :meth:`str.partition` element-wise", but it does not match str.partition's behavior for null bytes.

2. **Inconsistent with Python**: Python's `str.partition('\x00')` treats null bytes as valid single-character separators, but numpy.char incorrectly rejects them as "empty".

3. **Data corruption risk**: Null bytes are valid string characters in many contexts (binary protocols, null-terminated strings from C, etc.). Users expect them to be handled correctly.

4. **Same bug in rpartition**: Both `partition` and `rpartition` exhibit this bug.

## Fix

The bug appears to be in the separator validation logic. The functions should only reject truly empty separators (`''`), not null bytes (`'\x00'`).

```diff
--- a/numpy/_core/strings.py
+++ b/numpy/_core/strings.py
@@ -partition_function
-    if not sep:
+    if len(sep) == 0:
         raise ValueError("empty separator")
```

The current check likely uses a boolean test on the separator, which treats `'\x00'` as falsy. The fix should explicitly check the length instead.