# Bug Report: numpy.strings.slice Returns Empty String with None Stop

**Target**: `numpy.strings.slice`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`numpy.strings.slice()` returns an empty string when `stop=None` is explicitly passed, instead of slicing to the end of the string like Python's slice behavior.

## Property-Based Test

```python
import numpy as np
import numpy.strings as nps
from hypothesis import assume, given, strategies as st

@given(st.text(min_size=1), st.integers(min_value=0, max_value=20))
def test_slice_with_explicit_stop(s, start):
    assume(start < len(s))
    arr = np.array([s])
    result = nps.slice(arr, start, None)[0]
    expected = s[start:None]
    assert str(result) == expected
```

**Failing input**: `s='hello', start=0`

## Reproducing the Bug

```python
import numpy as np
import numpy.strings as nps

s = 'hello'
arr = np.array([s])
result = nps.slice(arr, 0, None)

print(f'Expected: {repr(s[0:None])}')
print(f'Got: {repr(result[0])}')
assert result[0] == s[0:None]
```

Output:
```
Expected: 'hello'
Got: ''
AssertionError
```

## Why This Is A Bug

The function's docstring states it should behave like Python's slice object. In Python, `s[0:None]` is equivalent to `s[0:]` which returns the string from index 0 to the end. However, `numpy.strings.slice(arr, 0, None)` returns an empty string.

The bug appears when `stop=None` is explicitly passed. The function should interpret `None` as "slice to the end" (like Python's default slice behavior), but instead appears to treat it as 0 or some other value.

Interestingly, when both start and stop are concrete integers, the function works correctly:
- `nps.slice(arr, 0, 5)` correctly returns 'hello'
- `nps.slice(arr, 1, 4)` correctly returns 'ell'

## Fix

The slice function needs to properly handle `None` as the stop parameter. In Python slice semantics, `None` means "go to the end". The implementation should check if `stop is None` and either:
1. Not pass it to the underlying slicing operation, or
2. Replace it with the string length

```diff
--- a/numpy/strings/slice.py
+++ b/numpy/strings/slice.py
@@ -someplace
-    if stop is not None:
-        result = _slice(arr, start, stop, step)
+    if stop is None:
+        stop = str_len(arr)
+    result = _slice(arr, start, stop, step)
```