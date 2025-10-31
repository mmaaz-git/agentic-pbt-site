# Bug Report: numpy.char.replace() Truncates When Result Expands

**Target**: `numpy.char.replace()` on chararray
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`numpy.char.replace()` silently truncates results when operating on a `chararray` and the replacement causes the string to expand beyond the original dtype size.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import numpy.char as char


@settings(max_examples=200)
@given(st.lists(st.text(min_size=1, max_size=5), min_size=1), st.text(min_size=1, max_size=2), st.text(min_size=1, max_size=5))
def test_replace_matches_python(strings, old, new):
    arr = char.array(strings)
    np_replaced = char.replace(arr, old, new)
    for i, s in enumerate(strings):
        py_replaced = s.replace(old, new)
        assert np_replaced[i] == py_replaced, f'{s!r}.replace({old!r}, {new!r}): numpy={np_replaced[i]!r}, python={py_replaced!r}'
```

**Failing input**: `strings=['0'], old='0', new='00'`

## Reproducing the Bug

```python
import numpy.char as char

arr = char.array(['0'])
result = char.replace(arr, '0', '00')

print(f"Input: '0'")
print(f"Expected after replace('0', '00'): '00'")
print(f"Got: {result[0]!r}")

assert result[0] == '00', f"Expected '00' but got {result[0]!r}"
```

Additional examples:
```python
import numpy.char as char

test_cases = [
    ('a', 'a', 'aa'),
    ('ab', 'b', 'bbb'),
    ('x', 'x', 'xyz'),
]

for s, old, new in test_cases:
    arr = char.array([s])
    result = char.replace(arr, old, new)
    expected = s.replace(old, new)
    match = result[0] == expected
    print(f"{s!r}.replace({old!r}, {new!r}): got {result[0]!r}, expected {expected!r}, match={match}")
```

## Why This Is A Bug

When `char.array()` creates a chararray, it infers the dtype from the input strings (e.g., `<U1` for single-character strings). When `char.replace()` operates on this chararray and the replacement expands the string, the result is truncated to fit the original dtype.

This violates the documented contract that `char.replace()` should behave like Python's `str.replace()`. The truncation is silent and produces fundamentally incorrect results.

The bug affects any replacement where `len(new) > len(old)` and the result would exceed the original dtype capacity.

Note: Using regular numpy arrays with adequate dtype size works correctly:
```python
import numpy as np

arr = np.array(['0'], dtype='U10')
result = char.replace(arr, '0', '00')
print(result[0])  # Correctly outputs '00'
```

## Fix

The fix requires `char.replace()` to compute an appropriate output dtype based on the potential string expansion:

```diff
--- a/numpy/_core/defchararray.py
+++ b/numpy/_core/defchararray.py
@@ -<line>
 def replace(a, old, new, count=-1):
     a_arr = np.asarray(a)
+    # Compute output dtype to accommodate potential expansion
+    if a_arr.dtype.kind == 'U':
+        # Worst case: every occurrence of 'old' is replaced with 'new'
+        if len(new) > len(old):
+            max_expansions = a_arr.dtype.itemsize // max(len(old), 1)
+            max_new_size = a_arr.dtype.itemsize + max_expansions * (len(new) - len(old))
+            out_dtype = f'U{max_new_size}'
+        else:
+            out_dtype = a_arr.dtype
+    else:
+        out_dtype = a_arr.dtype
-    return _replace(a_arr, old, new, count)
+    result = _replace(a_arr, old, new, count)
+    if out_dtype != a_arr.dtype:
+        return result.astype(out_dtype, copy=False)
+    return result
```