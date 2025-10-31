# Bug Report: Cython.Utils.strip_py2_long_suffix Empty String Crash

**Target**: `Cython.Utils.strip_py2_long_suffix`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`strip_py2_long_suffix` crashes with IndexError when given an empty string input.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from Cython.Utils import strip_py2_long_suffix

@given(st.text())
def test_empty_string_handling(value_str):
    result = strip_py2_long_suffix(value_str)
    assert isinstance(result, str)
```

**Failing input**: `value_str=''`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Utils import strip_py2_long_suffix

strip_py2_long_suffix("")
```

**Output**:
```
IndexError: string index out of range
```

## Why This Is A Bug

The function accesses `value_str[-1]` (Utils.py:468) without checking if the string is non-empty:

```python
def strip_py2_long_suffix(value_str):
    if value_str[-1] in 'lL':  # IndexError if value_str is empty
        return value_str[:-1]
    return value_str
```

While the function is typically called on `hex()` or `str()` output (which are never empty), defensive programming dictates that functions should handle edge cases gracefully rather than crash.

## Fix

```diff
--- a/Cython/Utils.py
+++ b/Cython/Utils.py
@@ -465,7 +465,7 @@ def strip_py2_long_suffix(value_str):
     Python 2 likes to append 'L' to stringified numbers
     which in then can't process when converting them to numbers.
     """
-    if value_str[-1] in 'lL':
+    if value_str and value_str[-1] in 'lL':
         return value_str[:-1]
     return value_str
```