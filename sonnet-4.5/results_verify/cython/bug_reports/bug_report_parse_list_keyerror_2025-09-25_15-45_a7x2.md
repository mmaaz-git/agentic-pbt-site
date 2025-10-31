# Bug Report: parse_list KeyError on Unclosed Quotes

**Target**: `Cython.Build.Dependencies.parse_list`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `parse_list` function crashes with a `KeyError` when given strings containing unclosed quotes (e.g., `"`, `'`, `""`, etc.). This happens because the `unquote` helper function incorrectly slices the literal label, removing the trailing underscore that is part of the label's format.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from Cython.Build.Dependencies import parse_list


@settings(max_examples=500)
@given(st.text())
def test_parse_list_returns_list(s):
    result = parse_list(s)
    assert isinstance(result, list), f"parse_list should return a list, got {type(result)}"
```

**Failing input**: `'"'` (single double-quote character)

## Reproducing the Bug

```python
from Cython.Build.Dependencies import parse_list

try:
    result = parse_list('"')
    print(f"Result: {result}")
except KeyError as e:
    print(f"KeyError: {e}")
```

**Output**:
```
KeyError: '__Pyx_L1'
```

**Other failing inputs**:
- `"'"` → `KeyError: '__Pyx_L1'`
- `'""'` → `KeyError: ''`
- `"''"` → `KeyError: ''`

## Why This Is A Bug

1. **Crash on valid-looking input**: While unclosed quotes may be malformed, the function should either:
   - Handle them gracefully
   - Raise a clear, descriptive error message
   - Not crash with an internal KeyError

2. **Root cause**: The `unquote` function at line 132 does:
   ```python
   return literals[literal[1:-1]]
   ```

   For input `'"__Pyx_L1_'`:
   - `literal[1:-1]` extracts `'__Pyx_L1'` (removes first and last char)
   - But the dictionary key is `'__Pyx_L1_'` (with trailing underscore)
   - This causes a KeyError

3. **Expected behavior**: The function should either:
   - Return an empty list for malformed input
   - Treat the unclosed quote as a literal string
   - Raise a clear `ValueError` explaining the issue

## Fix

The issue is in the `unquote` helper function within `parse_list`. The function needs to handle the case where the "content" of a quoted string is actually a label placeholder:

```diff
--- a/Cython/Build/Dependencies.py
+++ b/Cython/Build/Dependencies.py
@@ -128,9 +128,14 @@ def parse_list(s):
     s, literals = strip_string_literals(s)
     def unquote(literal):
         literal = literal.strip()
-        if literal[0] in "'\"":
-            return literals[literal[1:-1]]
-        else:
+        if literal and literal[0] in "'\"":
+            # Extract the content between the quotes
+            content = literal[1:-1]
+            # Check if it's a label placeholder
+            if content in literals:
+                return literals[content]
+            # If not found, it might be the label was truncated, try with quotes
+            return literals.get(literal, literal)
+        elif literal:
             return literal
     return [unquote(item) for item in s.split(delimiter) if item.strip()]
```

However, a simpler fix that aligns with the intended behavior is to recognize that when `strip_string_literals` encounters unclosed quotes, it creates a label that includes the quote character. The `unquote` function should look for the full label including quotes:

```diff
--- a/Cython.Build.Dependencies.py
+++ b/Cython.Build.Dependencies.py
@@ -128,7 +128,10 @@ def parse_list(s):
     s, literals = strip_string_literals(s)
     def unquote(literal):
         literal = literal.strip()
-        if literal[0] in "'\"":
+        if not literal:
+            return literal
+        # Check if this looks like a quoted placeholder
+        if literal[0] in "'\"" and len(literal) > 2 and literal[-1] in "'\"":
             return literals[literal[1:-1]]
         else:
             return literal
```

This prevents the function from trying to unquote strings that don't have both opening and closing quotes.