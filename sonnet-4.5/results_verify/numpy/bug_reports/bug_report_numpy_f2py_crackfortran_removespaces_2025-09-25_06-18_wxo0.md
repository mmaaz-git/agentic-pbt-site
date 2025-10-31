# Bug Report: numpy.f2py.crackfortran.removespaces strips all whitespace

**Target**: `numpy.f2py.crackfortran.removespaces`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `removespaces` function incorrectly removes all leading/trailing whitespace characters (newlines, tabs, carriage returns), not just spaces, due to using `str.strip()` before processing.

## Property-Based Test

```python
from hypothesis import given, settings
from hypothesis import strategies as st
import numpy.f2py.crackfortran as cf


@given(st.text(min_size=1))
@settings(max_examples=1000)
def test_removespaces_preserves_non_space_whitespace(text):
    result = cf.removespaces(text)
    expected = text.replace(' ', '')

    for char in ['\n', '\r', '\t']:
        if char in text and char not in ' ':
            if char in expected and char not in result:
                assert False, f"removespaces removed {repr(char)} which is not a space"
```

**Failing input**: `text='\n'`

## Reproducing the Bug

```python
import numpy.f2py.crackfortran as cf

print(repr(cf.removespaces('\n')))
print(repr(cf.removespaces('\ra\r')))
print(repr(cf.removespaces('\t')))
```

Expected output: `'\n'`, `'\ra\r'`, `'\t'`

Actual output: `''`, `'a'`, `''`

## Why This Is A Bug

The function is named `removespaces`, implying it should only remove space characters (ASCII 32). However, it calls `str.strip()` which removes all whitespace characters including newlines (`\n`), carriage returns (`\r`), and tabs (`\t`). This violates the principle of least surprise. While this is unlikely to cause issues in typical Fortran parsing (as these characters are usually line separators), the function name and behavior are mismatched.

## Fix

```diff
--- a/numpy/f2py/crackfortran.py
+++ b/numpy/f2py/crackfortran.py
@@ -xxx,7 +xxx,7 @@ def removespaces(expr):
 def removespaces(expr):
-    expr = expr.strip()
+    expr = expr.strip(' ')
     if len(expr) <= 1:
         return expr
     expr2 = expr[0]
```