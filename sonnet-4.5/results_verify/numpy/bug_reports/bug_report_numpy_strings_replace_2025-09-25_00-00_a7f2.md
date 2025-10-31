# Bug Report: numpy.strings.replace Truncates Replacement String

**Target**: `numpy.strings.replace`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When `replace()` is called with scalar string arguments for `old` and `new`, the `new` string is incorrectly truncated to the input array's dtype width before buffer size calculation, causing silent data loss when the replacement string is longer than the original array's element width.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings, assume
import numpy as np
import numpy.strings as nps

@given(st.text(min_size=1, max_size=20).filter(lambda x: '\x00' not in x),
       st.text(min_size=1, max_size=20).filter(lambda x: '\x00' not in x),
       st.text(max_size=10).filter(lambda x: '\x00' not in x))
@settings(max_examples=500)
def test_replace_matches_python(prefix, old, suffix):
    assume(len(old) > 0)
    assume(len(prefix) + len(old) + len(suffix) < 50)

    s = prefix + old + suffix
    new = 'X' * (len(old) + 5) if len(old) < 10 else 'Y'

    arr = np.array([s])
    result = nps.replace(arr, old, new, count=1)
    python_result = s.replace(old, new, 1)

    assert result[0] == python_result
```

**Failing input**: `s='00', old='0', new='XXXXXX', count=1`

## Reproducing the Bug

```python
import numpy as np
import numpy.strings as nps

s = '00'
arr = np.array([s])
result = nps.replace(arr, '0', 'XXXXXX', count=1)

print(f"Input: {s!r}")
print(f"Python result: {s.replace('0', 'XXXXXX', 1)!r}")
print(f"NumPy result:  {result[0]!r}")

assert result[0] == 'XXXXXX0'
```

Output:
```
Input: '00'
Python result: 'XXXXXX0'
NumPy result:  'XX0'
AssertionError
```

## Why This Is A Bug

The function's docstring states it performs string replacement element-wise, mimicking Python's `str.replace()`. However, when scalar string arguments are passed for `old` and `new` parameters, the function incorrectly casts `new` to the input array's dtype BEFORE calculating the required buffer size.

For input `arr=['00']` (dtype `<U2`), when `new='XXXXXX'` is cast to `<U2`, it becomes `'XX'` (truncated to 2 characters). The subsequent buffer size calculation uses this truncated value:
- `buffersizes = str_len(arr) + counts * (str_len(new) - str_len(old))`
- `buffersizes = 2 + 1 * (2 - 1) = 3` (should be `2 + 1 * (6 - 1) = 7`)

This creates an output array with dtype `<U3`, which silently truncates the actual replacement result from `'XXXXXX0'` to `'XX0'`.

## Fix

The bug is in lines 1358-1359 of `numpy/_core/strings.py`. The `new` string should NOT be cast to the input array's dtype before buffer size calculation. Instead, calculate buffer size using the original length of `new`, then cast only for the actual replacement operation.

```diff
--- a/numpy/_core/strings.py
+++ b/numpy/_core/strings.py
@@ -1355,13 +1355,14 @@ def replace(a, old, new, count=-1):
         return _replace(arr, old, new, count)

     a_dt = arr.dtype
-    old = old.astype(old_dtype or a_dt, copy=False)
-    new = new.astype(new_dtype or a_dt, copy=False)
     max_int64 = np.iinfo(np.int64).max
     counts = _count_ufunc(arr, old, 0, max_int64)
     counts = np.where(count < 0, counts, np.minimum(counts, count))
     buffersizes = str_len(arr) + counts * (str_len(new) - str_len(old))
     out_dtype = f"{arr.dtype.char}{buffersizes.max()}"
     out = np.empty_like(arr, shape=buffersizes.shape, dtype=out_dtype)
+
+    old = old.astype(old_dtype or a_dt, copy=False)
+    new = new.astype(new_dtype or a_dt, copy=False)

     return _replace(arr, old, new, counts, out=out)
```