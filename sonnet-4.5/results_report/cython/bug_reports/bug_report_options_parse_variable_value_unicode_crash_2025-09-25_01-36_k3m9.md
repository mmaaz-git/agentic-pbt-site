# Bug Report: Options.parse_variable_value Unicode Digit Crash

**Target**: `Cython.Compiler.Options.parse_variable_value`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `parse_variable_value` function crashes with ValueError when given Unicode digit characters because it uses `str.isdigit()` to check if a string is numeric, but `int()` doesn't accept Unicode digits.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from Cython.Compiler import Options

@given(st.text(min_size=1))
def test_parse_variable_value_no_crash(s):
    result = Options.parse_variable_value(s)
```

**Failing input**: `'²'` (superscript 2, or any Unicode digit)

## Reproducing the Bug

```python
from Cython.Compiler import Options

result = Options.parse_variable_value('²')
```

Output:
```
ValueError: invalid literal for int() with base 10: '²'
```

## Why This Is A Bug

The function uses `value.isdigit()` (line 609) to check if a value should be parsed as an integer. However, Python's `str.isdigit()` returns `True` for Unicode digit characters like '²' (superscript 2), '①' (circled one), '೧' (Kannada digit), etc. When the code then calls `int(value)` (line 610), it crashes because `int()` only accepts ASCII digits 0-9.

This function is used by `parse_compile_time_env()` which is called from command-line parsing in `CmdLine.py`. If a user provides a Unicode digit in a compile-time environment variable, Cython will crash.

## Fix

Replace `isdigit()` with `isdecimal()` which only returns True for ASCII digits:

```diff
--- a/Cython/Compiler/Options.py
+++ b/Cython/Compiler/Options.py
@@ -606,7 +606,7 @@ def parse_variable_value(value):
         return False
     elif value == "None":
         return None
-    elif value.isdigit():
+    elif value.isdecimal() and value.isascii():
         return int(value)
     else:
         try:
```

Alternatively, use defensive exception handling:

```diff
--- a/Cython/Compiler/Options.py
+++ b/Cython/Compiler/Options.py
@@ -606,10 +606,13 @@ def parse_variable_value(value):
         return False
     elif value == "None":
         return None
-    elif value.isdigit():
+    else:
+        try:
+            return int(value)
+        except ValueError:
+            pass
+        try:
-        return int(value)
-    else:
-        try:
             value = float(value)
         except Exception:
```