# Bug Report: numpy.char.str_len Truncates at Trailing Null Characters

**Target**: `numpy.char.str_len`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`numpy.char.str_len()` incorrectly treats trailing null characters (`\x00`) as C-style string terminators, returning incorrect string lengths. This contradicts Python's string semantics where `\x00` is a valid character.

## Property-Based Test

```python
import numpy.char as char
import numpy as np
from hypothesis import given, strategies as st, settings


@given(st.text(min_size=0, max_size=20))
@settings(max_examples=1000)
def test_str_len_matches_python_len(s):
    arr = np.array([s])
    numpy_len = char.str_len(arr)[0]
    python_len = len(s)
    assert numpy_len == python_len
```

**Failing input**: `'\x00'`

## Reproducing the Bug

```python
import numpy.char as char
import numpy as np

test_cases = ['\x00', 'a\x00', 'abc\x00', '\x00\x00']

for s in test_cases:
    arr = np.array([s])
    numpy_len = char.str_len(arr)[0]
    python_len = len(s)
    print(f'{s!r:15} | Python len: {python_len} | numpy str_len: {numpy_len}')
```

Output:
```
'\x00'          | Python len: 1 | numpy str_len: 0
'a\x00'         | Python len: 2 | numpy str_len: 1
'abc\x00'       | Python len: 4 | numpy str_len: 3
'\x00\x00'      | Python len: 2 | numpy str_len: 0
```

## Why This Is A Bug

In Python, `\x00` is a valid character in Unicode strings. However, `numpy.char.str_len()` uses C-style string length calculation that stops at null terminators.

The function correctly handles `\x00` when followed by other characters (e.g., `'abc\x00def'` → length 7), but fails when `\x00` appears at the end of the string.

This causes:
1. Incorrect string length calculations for strings ending with `\x00`
2. Validation failures in `center`, `ljust`, and `rjust` when using `\x00` as fillchar
3. Potential incorrect behavior in functions relying on `str_len`

## Fix

The bug is in the C implementation of `numpy._core.umath.str_len`. The function needs to use Python's string length directly rather than scanning for null terminators:

```diff
--- a/numpy/_core/src/umath/string_ufuncs.c
+++ b/numpy/_core/src/umath/string_ufuncs.c
@@ -xxx,x +xxx,x @@
-    /* Scan for null terminator */
-    while (len < max_len && str[len] != '\0') {
-        len++;
-    }
+    /* Use Python's Unicode string length */
+    len = PyUnicode_GET_LENGTH(str_obj);
```