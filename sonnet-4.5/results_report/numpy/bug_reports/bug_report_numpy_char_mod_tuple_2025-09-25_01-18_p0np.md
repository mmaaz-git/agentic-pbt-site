# Bug Report: numpy.char.mod Tuple Argument Handling

**Target**: `numpy.char.mod`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`numpy.char.mod` fails to handle tuple arguments for format strings with multiple placeholders, while Python's built-in `%` operator handles them correctly.

## Property-Based Test

```python
import numpy as np
import numpy.char as char
from hypothesis import given, strategies as st

@given(st.lists(st.integers(min_value=0, max_value=100), min_size=1))
def test_mod_with_multiple_formats(values):
    format_strings = ['value: %d, hex: %x'] * len(values)
    arr = np.array(format_strings, dtype=str)

    for i in range(len(values)):
        result = char.mod(arr[i:i+1], (values[i], values[i]))
        expected = 'value: %d, hex: %x' % (values[i], values[i])
        assert result[0] == expected
```

**Failing input**: `values=[0]` (or any integer)

## Reproducing the Bug

```python
import numpy as np
import numpy.char as char

arr = np.array(['x=%d, y=%d'], dtype=str)
result = char.mod(arr, (5, 10))
```

Output:
```
TypeError: not enough arguments for format string
```

Expected:
```python
result[0] == 'x=5, y=10'
```

Comparison with Python's `%` operator:
```python
'x=%d, y=%d' % (5, 10)
```

This correctly produces: `'x=5, y=10'`

## Why This Is A Bug

The `char.mod` function is documented as element-wise string formatting using the `%` operator. Python's `%` operator accepts tuple arguments for format strings with multiple placeholders, so `char.mod` should too. Currently, dict-based formatting works (e.g., `'%(name)s'`), but tuple-based formatting fails.

## Fix

The issue is in `/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/strings.py:268`:

```python
_vec_string(a, np.object_, '__mod__', (values,)), a)
```

The `values` argument is wrapped in a tuple `(values,)`, which causes `'x=%d, y=%d'.__mod__((5, 10))` to become `'x=%d, y=%d'.__mod__(((5, 10),))`, adding an extra tuple layer.

The fix would be to check if `values` is already a tuple and handle it appropriately:

```diff
--- a/numpy/_core/strings.py
+++ b/numpy/_core/strings.py
@@ -265,7 +265,10 @@ def mod(a, values):
     a = np.asanyarray(a)
     if not issubclass(a.dtype.type, (np.bytes_, np.str_)):
         raise TypeError("unsupported type for operand 'a'")
-    return _vec_string(a, np.object_, '__mod__', (values,)), a)
+    if isinstance(values, tuple):
+        return _to_string_or_unicode_array(_vec_string(a, np.object_, '__mod__', values), a)
+    else:
+        return _to_string_or_unicode_array(_vec_string(a, np.object_, '__mod__', (values,)), a)
```