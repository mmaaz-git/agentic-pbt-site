# Bug Report: numpy.char.str_len Truncates at Trailing Null Characters

**Target**: `numpy.char.str_len`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`numpy.char.str_len()` incorrectly treats null characters (`\x00`) at the end of strings as C-style string terminators, returning incorrect string lengths. This contradicts Python's string semantics where `\x00` is a valid character.

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

test_cases = [
    ('\x00', 1),
    ('a\x00', 2),
    ('abc\x00', 4),
    ('\x00\x00', 2),
]

for s, expected in test_cases:
    arr = np.array([s])
    numpy_len = char.str_len(arr)[0]
    python_len = len(s)
    print(f'{s!r:15} | Python: {python_len} | numpy: {numpy_len} | Expected: {expected}')
```

Output:
```
'\x00'          | Python: 1 | numpy: 0 | Expected: 1
'a\x00'         | Python: 2 | numpy: 1 | Expected: 2
'abc\x00'       | Python: 4 | numpy: 3 | Expected: 4
'\x00\x00'      | Python: 2 | numpy: 0 | Expected: 2
```

## Why This Is A Bug

In Python, `\x00` is a valid character in Unicode strings. Python's `len('\x00')` correctly returns 1. However, `numpy.char.str_len()` uses C-style string length calculation that stops at null terminators, leading to incorrect results when strings contain trailing `\x00` characters.

Notably, `str_len()` correctly handles `\x00` when followed by other characters (e.g., `'abc\x00def'`), but fails when `\x00` appears at the end.

This bug causes downstream issues:
1. Incorrect string length calculations for any string ending with `\x00`
2. Validation failures in `center`, `ljust`, and `rjust` when using `\x00` as fillchar (error: "The fill character must be exactly one character long")
3. Potential incorrect behavior in other functions that rely on `str_len` internally
4. Breaks processing of binary data or special encodings stored in string arrays

## Fix

The bug is in the C implementation of `numpy._core.umath.str_len`. The function appears to be scanning backwards from the end of the string and stopping when it encounters a non-null character, similar to C's `strlen()` behavior.

The fix requires modifying the C code to use Python's string length directly rather than scanning for null terminators. Python strings already know their length and can contain `\x00` as valid characters.

Suggested approach:
```diff
--- a/numpy/_core/src/umath/string_ufuncs.c
+++ b/numpy/_core/src/umath/string_ufuncs.c
@@ -xxx,x +xxx,x @@
-    /* Current implementation likely uses strlen-like logic */
-    size_t len = 0;
-    while (len < max_len && str[len] != '\0') {
-        len++;
-    }
-    return len;
+    /* Use Python's string length directly */
+    return PyUnicode_GET_LENGTH(str_obj);
```

The exact fix depends on the actual C implementation details, but the principle is to use Python's Unicode string length rather than scanning for null terminators.