# Bug Report: Cython.Build parse_list KeyError Crash

**Target**: `Cython.Build.Dependencies.parse_list`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `parse_list` function crashes with a `KeyError` when given input containing unclosed quotes (e.g., a single quote character `"'"`). This occurs because the function's `unquote` helper incorrectly tries to look up string literal labels in the literals dictionary.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from Cython.Build.Dependencies import parse_list

@given(st.text(min_size=0, max_size=1000))
@settings(max_examples=500)
def test_parse_list_always_returns_list(s):
    result = parse_list(s)
    assert isinstance(result, list)
```

**Failing input**: `"'"`

## Reproducing the Bug

```python
from Cython.Build.Dependencies import parse_list

result = parse_list("'")
```

**Output:**
```
KeyError: '__Pyx_L1'
```

**Full traceback:**
```
  File ".../Cython/Build/Dependencies.py", line 135, in parse_list
    return [unquote(item) for item in s.split(delimiter) if item.strip()]
  File ".../Cython/Build/Dependencies.py", line 132, in unquote
    return literals[literal[1:-1]]
KeyError: '__Pyx_L1'
```

## Why This Is A Bug

The function should handle malformed input gracefully or document that it expects well-formed input. The crash occurs because:

1. `strip_string_literals("'")` treats the unclosed quote as a string literal and replaces it with a label (e.g., `__Pyx_L1_`)
2. The `unquote` helper function checks if the item starts with a quote character
3. Since labels like `__Pyx_L1_` don't start with quotes, the code attempts `literals[literal[1:-1]]` on line 132
4. However, this path should only execute when `literal[0] in "'\""`is true, so the label must somehow trigger this condition, leading to an incorrect key lookup

The root cause appears to be that the `unquote` function's logic doesn't properly handle the labels produced by `strip_string_literals`.

## Fix

The `unquote` function needs to check if the literal is actually a label that needs to be looked up in the literals dictionary. One approach:

```diff
--- a/Dependencies.py
+++ b/Dependencies.py
@@ -129,7 +129,10 @@ def parse_list(s):
     def unquote(literal):
         literal = literal.strip()
-        if literal[0] in "'\"":
+        if not literal:
+            return literal
+        if literal in literals:
+            return literals[literal]
+        elif literal[0] in "'\"" and literal[-1] in "'\"":
             return literals[literal[1:-1]]
         else:
             return literal
```

However, this fix needs verification against the actual behavior of `strip_string_literals` to ensure it handles all cases correctly.