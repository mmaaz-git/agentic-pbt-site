# Bug Report: Options.parse_variable_value Large Integer Precision Loss

**Target**: `Cython.Compiler.Options.parse_variable_value`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `parse_variable_value` function silently loses precision when parsing large integers by converting them to floats, corrupting compile-time integer constants.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from Cython.Compiler import Options

@given(st.integers())
def test_parse_variable_value_preserves_integers(n):
    s = str(n)
    result = Options.parse_variable_value(s)
    if s.lstrip('-').isdigit():
        assert result == n
```

**Failing input**: `-9007199254740993`

## Reproducing the Bug

```python
from Cython.Compiler import Options

large_int = -9007199254740993
result = Options.parse_variable_value(str(large_int))

print(f"Input: {large_int}")
print(f"Result: {result}")
print(f"Type: {type(result)}")
print(f"Equal: {result == large_int}")
```

Output:
```
Input: -9007199254740993
Result: -9007199254740992.0
Type: <class 'float'>
Equal: False
```

## Why This Is A Bug

The function is designed to parse compile-time environment variables, which can be integers, floats, booleans, or strings. When given an integer string, it should return an exact `int` value.

The bug occurs for negative integers:
1. Line 609: `value.isdigit()` returns `False` for negative integers (due to '-' sign)
2. Code falls through to line 613: `float(value)`
3. For large integers beyond float precision (~15-16 digits), `float()` loses precision
4. The integer `-9007199254740993` becomes `-9007199254740992.0` (off by 1)

This is used by `parse_compile_time_env()` called from `CmdLine.py`, so users can trigger this via:
```bash
cython -E LARGE_CONSTANT=-9007199254740993 file.pyx
```

The constant will be silently corrupted in the compiled code.

## Fix

Try parsing as `int` before falling back to `float` to preserve integer precision:

```diff
--- a/Cython/Compiler/Options.py
+++ b/Cython/Compiler/Options.py
@@ -606,11 +606,15 @@ def parse_variable_value(value):
         return False
     elif value == "None":
         return None
-    elif value.isdigit():
-        return int(value)
     else:
+        try:
+            return int(value)
+        except ValueError:
+            pass
         try:
-            value = float(value)
+            return float(value)
         except Exception:
             pass
-        return value
+        return value
```