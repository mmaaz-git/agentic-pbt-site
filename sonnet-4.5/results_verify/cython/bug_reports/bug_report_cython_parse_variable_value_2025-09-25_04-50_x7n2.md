# Bug Report: Cython.Compiler.Options.parse_variable_value Crashes on Unicode Digits

**Target**: `Cython.Compiler.Options.parse_variable_value`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `parse_variable_value` function crashes with `ValueError` when given Unicode digit characters like '²' because it uses `str.isdigit()` (which returns True for Unicode digits) but then calls `int()` (which only accepts ASCII digits).

## Property-Based Test

```python
from hypothesis import given, strategies as st
from Cython.Compiler.Options import parse_variable_value


@given(st.text())
def test_parse_variable_value_no_crash(s):
    result = parse_variable_value(s)
    assert result is not None or s == "None"
```

**Failing input**: `s='²'`

## Reproducing the Bug

```python
from Cython.Compiler.Options import parse_variable_value

result = parse_variable_value('²')
```

Output:
```
ValueError: invalid literal for int() with base 10: '²'
```

## Why This Is A Bug

The function uses `value.isdigit()` to check if the input is a number, but `isdigit()` returns `True` for Unicode digit characters like superscript digits ('²', '³', '¹') and digits from other scripts. However, `int()` only accepts ASCII digits, causing a crash.

This violates the expected behavior for a parser function, which should handle invalid inputs gracefully rather than raising unhandled exceptions.

## Fix

```diff
--- a/Options.py
+++ b/Options.py
@@ -1,7 +1,7 @@
     elif value == "False":
         return False
     elif value == "None":
         return None
-    elif value.isdigit():
+    elif value.isdecimal():
         return int(value)
     else:
```

Alternatively, wrap the `int()` call in a try-except to handle the ValueError gracefully.