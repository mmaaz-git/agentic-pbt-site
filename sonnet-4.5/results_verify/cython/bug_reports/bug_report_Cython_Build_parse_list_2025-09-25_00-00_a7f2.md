# Bug Report: Cython.Build.Dependencies.parse_list KeyError on Quote Characters

**Target**: `Cython.Build.Dependencies.parse_list`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `parse_list` function crashes with a `KeyError` when parsing strings containing unclosed quote characters (single or double quotes) or certain other special characters like `#`. This is due to incorrect string slicing logic when looking up normalized string literals in the internal literals dictionary.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from Cython.Build.Dependencies import parse_list

@given(st.text(alphabet=st.characters(min_codepoint=32, max_codepoint=126), min_size=0, max_size=100))
def test_parse_list_no_crash(s):
    result = parse_list(s)
    assert isinstance(result, list)
```

**Failing input**: `"'"` (single quote character)

## Reproducing the Bug

```python
from Cython.Build.Dependencies import parse_list

parse_list("'")
```

Output:
```
KeyError: '__Pyx_L1'
```

Additional failing inputs:
```python
parse_list('"')
parse_list("0'")
parse_list("#")
parse_list("[']")
```

## Why This Is A Bug

The `parse_list` function is documented to parse list-like strings and handle quoted elements (see doctests in lines 110-122). However, it fails on legitimate inputs containing quote characters. While unclosed quotes might indicate malformed input, the function should either handle them gracefully or raise a more meaningful error rather than crashing with a `KeyError` about an internal implementation detail (`__Pyx_L1`).

The root cause is in the `unquote` helper function (lines 129-134):

```python
def unquote(literal):
    literal = literal.strip()
    if literal[0] in "'\"":
        return literals[literal[1:-1]]  # Bug: literal[1:-1] removes trailing underscore
    else:
        return literal
```

When `strip_string_literals` normalizes the input `"'"`, it produces:
- `normalized = "'__Pyx_L1_"` (quote followed by label)
- `literals = {"__Pyx_L1_": ""}`

The `unquote` function then does `literal[1:-1]` on `"'__Pyx_L1_"`, yielding `"__Pyx_L1"` (without trailing underscore), but the actual key in the dictionary is `"__Pyx_L1_"` (with trailing underscore).

## Fix

```diff
--- a/Cython/Build/Dependencies.py
+++ b/Cython/Build/Dependencies.py
@@ -129,7 +129,10 @@ def parse_list(s):
     def unquote(literal):
         literal = literal.strip()
-        if literal[0] in "'\"":
-            return literals[literal[1:-1]]
+        if literal and literal[0] in "'\"":
+            key = literal[1:-1]
+            if key in literals:
+                return literals[key]
+            return literal
         else:
             return literal
     return [unquote(item) for item in s.split(delimiter) if item.strip()]
```

This fix adds bounds checking and gracefully handles the case where the key is not found in the literals dictionary by returning the literal itself unchanged.