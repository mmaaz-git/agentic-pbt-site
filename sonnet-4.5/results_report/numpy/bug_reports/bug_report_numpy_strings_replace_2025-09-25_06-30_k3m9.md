# Bug Report: numpy.strings.replace Truncates Replacement String

**Target**: `numpy.strings.replace`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `numpy.strings.replace` function incorrectly truncates the replacement string to match the input array's dtype size, causing incorrect results when the replacement is longer than the original array's maximum string length.

## Property-Based Test

```python
import numpy as np
import numpy.strings as nps
from hypothesis import given, strategies as st, assume


def string_arrays():
    return st.lists(
        st.text(
            alphabet=st.characters(min_codepoint=32, max_codepoint=126),
            min_size=0, max_size=30
        ),
        min_size=1, max_size=20
    ).map(lambda lst: np.array(lst, dtype='U'))


def simple_strings():
    return st.text(
        alphabet=st.characters(min_codepoint=32, max_codepoint=126),
        min_size=0, max_size=20
    )


@given(string_arrays(), simple_strings(), simple_strings(), st.integers(min_value=0, max_value=10))
def test_replace_matches_python_semantics(arr, old_str, new_str, count):
    assume(old_str != '')

    result = nps.replace(arr, old_str, new_str, count=count)

    for i in range(len(arr)):
        expected = str(arr[i]).replace(old_str, new_str, count)
        actual = str(result[i])
        assert actual == expected
```

**Failing input**: `arr=['0']` (dtype `<U1`), `old_str='0'`, `new_str='00'`, `count=1`

## Reproducing the Bug

```python
import numpy as np
import numpy.strings as nps

arr = np.array(['0'])
result = nps.replace(arr, '0', '00', count=1)
print(f"Result: {result}")
print(f"Expected: ['00']")
print(f"Bug: {str(result[0]) != '00'}")
```

Output:
```
Result: ['0']
Expected: ['00']
Bug: True
```

The bug occurs because:
1. Input array has dtype `<U1` (max 1 character)
2. Replacement string '00' is cast to `<U1`, truncating it to '0'
3. Buffer size is calculated as: `str_len(arr) + count * (str_len('0') - str_len('0'))` = `1 + 1 * 0` = `1`
4. Result array is created with dtype `<U1`, which cannot hold '00'

## Why This Is A Bug

This violates the documented behavior of `numpy.strings.replace`, which should mimic Python's `str.replace()`. The function produces incorrect results for legitimate use cases where the replacement string is longer than the input strings.

## Fix

The issue is in lines 1358-1359 of `numpy/_core/strings.py`:

```python
a_dt = arr.dtype
old = old.astype(old_dtype or a_dt, copy=False)
new = new.astype(new_dtype or a_dt, copy=False)
```

The replacement string should not be truncated to the input array's dtype. The fix is to calculate `str_len(new)` and `str_len(old)` before casting, or preserve their original length:

```diff
--- a/numpy/_core/strings.py
+++ b/numpy/_core/strings.py
@@ -1355,11 +1355,13 @@ def replace(a, old, new, count=-1):
         return _replace(arr, old, new, count)

     a_dt = arr.dtype
-    old = old.astype(old_dtype or a_dt, copy=False)
-    new = new.astype(new_dtype or a_dt, copy=False)
+    old_len = str_len(old)
+    new_len = str_len(new)
     max_int64 = np.iinfo(np.int64).max
     counts = _count_ufunc(arr, old, 0, max_int64)
     counts = np.where(count < 0, counts, np.minimum(counts, count))
-    buffersizes = str_len(arr) + counts * (str_len(new) - str_len(old))
+    old = old.astype(old_dtype or a_dt, copy=False)
+    new = new.astype(new_dtype or a_dt, copy=False)
+    buffersizes = str_len(arr) + counts * (new_len - old_len)
     out_dtype = f"{arr.dtype.char}{buffersizes.max()}"
     out = np.empty_like(arr, shape=buffersizes.shape, dtype=out_dtype)
```