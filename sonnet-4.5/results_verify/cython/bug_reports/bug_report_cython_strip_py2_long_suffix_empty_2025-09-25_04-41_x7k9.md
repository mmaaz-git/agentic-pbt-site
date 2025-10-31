# Bug Report: Cython.Utils.strip_py2_long_suffix IndexError on Empty String

**Target**: `Cython.Utils.strip_py2_long_suffix`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `strip_py2_long_suffix` function crashes with an `IndexError` when given an empty string as input, due to unchecked access to `value_str[-1]`.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from Cython.Utils import strip_py2_long_suffix

@given(st.text(min_size=0, max_size=100))
def test_strip_py2_long_suffix_idempotence(s):
    result1 = strip_py2_long_suffix(s)
    result2 = strip_py2_long_suffix(result1)
    assert result1 == result2
```

**Failing input**: `''` (empty string)

## Reproducing the Bug

```python
from Cython.Utils import strip_py2_long_suffix

strip_py2_long_suffix('')
```

Output:
```
IndexError: string index out of range
```

## Why This Is A Bug

The function `strip_py2_long_suffix` is a public function (not prefixed with `_`) that performs string manipulation. It should handle all string inputs gracefully, including the edge case of an empty string. The function unconditionally accesses `value_str[-1]` at line 468 without first checking if the string is non-empty, causing a crash on empty input.

While the current caller (`str_to_number` at line 447) ensures the string is non-empty, this function is exported from the module and could be called from other contexts where empty strings are valid inputs.

## Fix

```diff
--- a/Cython/Utils.py
+++ b/Cython/Utils.py
@@ -465,6 +465,8 @@ def strip_py2_long_suffix(value_str):
     Python 2 likes to append 'L' to stringified numbers
     which in then can't process when converting them to numbers.
     """
+    if not value_str:
+        return value_str
     if value_str[-1] in 'lL':
         return value_str[:-1]
     return value_str
```