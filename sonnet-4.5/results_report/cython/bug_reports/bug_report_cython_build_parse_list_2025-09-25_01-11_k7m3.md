# Bug Report: Cython.Build.Dependencies.parse_list KeyError on Unclosed Quotes

**Target**: `Cython.Build.Dependencies.parse_list`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`parse_list` crashes with a `KeyError` when given strings containing unclosed or malformed quotes, such as a single `"` character.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from Cython.Build.Dependencies import parse_list


@given(st.text())
@settings(max_examples=1000)
def test_parse_list_should_not_crash(s):
    result = parse_list(s)
```

**Failing input**: `'"'` (a single double-quote character)

## Reproducing the Bug

```python
from Cython.Build.Dependencies import parse_list

parse_list('"')
```

Output:
```
KeyError: '__Pyx_L1'
```

## Why This Is A Bug

The function crashes on simple string inputs containing unclosed quotes. The issue occurs because:

1. `strip_string_literals('"')` returns `('"__Pyx_L1_', {'__Pyx_L1_': ''})`
2. The internal `unquote` function sees the result starts with `"` and tries to look up `literal[1:-1]` = `'__Pyx_L1'` in the literals dict
3. But the actual key is `'__Pyx_L1_'` (with trailing underscore)
4. This mismatch causes a `KeyError`

The function provides no documentation that it requires well-formed quoted strings, and crashes instead of handling malformed input gracefully or raising an informative error.

## Fix

```diff
--- a/Cython/Build/Dependencies.py
+++ b/Cython/Build/Dependencies.py
@@ -129,7 +129,10 @@ def parse_list(s):
     s, literals = strip_string_literals(s)
     def unquote(literal):
         literal = literal.strip()
-        if literal[0] in "'\"":
+        if len(literal) >= 2 and literal[0] in "'\"" and literal[-1] in "'\"":
+            return literals.get(literal[1:-1], literal)
+        elif literal[0] in "'\"":
+            # Malformed quoted string, return as-is
             return literals[literal[1:-1]]
         else:
             return literal
```

Alternatively, use `.get()` with a fallback to handle missing keys gracefully:

```diff
--- a/Cython/Build/Dependencies.py
+++ b/Cython/Build/Dependencies.py
@@ -129,7 +129,7 @@ def parse_list(s):
     s, literals = strip_string_literals(s)
     def unquote(literal):
         literal = literal.strip()
         if literal[0] in "'\"":
-            return literals[literal[1:-1]]
+            return literals.get(literal[1:-1], literal)
         else:
             return literal
```