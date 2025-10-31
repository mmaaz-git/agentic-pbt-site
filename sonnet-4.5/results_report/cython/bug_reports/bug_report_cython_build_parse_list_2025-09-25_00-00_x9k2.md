# Bug Report: Cython.Build.Dependencies.parse_list KeyError on Unclosed Quotes

**Target**: `Cython.Build.Dependencies.parse_list`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `parse_list` function crashes with a `KeyError` when given input containing unclosed or lone quote characters, despite being used to parse user-written distutils directive values that could contain such malformed input.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from Cython.Build.Dependencies import parse_list

@given(st.text())
@settings(max_examples=1000)
def test_parse_list_no_empty_strings(s):
    result = parse_list(s)
    assert all(item != '' for item in result), f"parse_list returned empty string in result: {result}"
```

**Failing input**: `"'"`

## Reproducing the Bug

```python
from Cython.Build.Dependencies import parse_list

parse_list("'")
```

Output:
```
KeyError: '__Pyx_L1'
```

This also fails with `"` (double quote) and `"'hello"` (unclosed quote).

## Why This Is A Bug

The `parse_list` function is used to parse distutils directive values from comments in user .pyx/.py files (see `DistutilsInfo.__init__` in Dependencies.py:197). Users manually write these comments and could easily introduce typos with unclosed quotes, for example:

```python
# distutils: libraries = ['ssl
```

Instead of gracefully handling or reporting malformed input, the function crashes with a cryptic `KeyError` that doesn't help users understand the actual problem (unclosed quote). At minimum, it should raise a more descriptive error.

## Fix

The bug occurs in the `unquote` nested function within `parse_list`. When given an unclosed quote like `'`, `strip_string_literals` normalizes it to `'__Pyx_L1_` and stores the key `__Pyx_L1_` in the literals dict. However, `unquote` tries to look up `literal[1:-1]` which evaluates to `__Pyx_L1` (removing both the leading quote and the trailing underscore). This key doesn't exist in the dict, causing a KeyError.

```diff
--- a/Cython/Build/Dependencies.py
+++ b/Cython/Build/Dependencies.py
@@ -129,7 +129,12 @@ def parse_list(s):
     s, literals = strip_string_literals(s)
     def unquote(literal):
         literal = literal.strip()
         if literal[0] in "'\"":
-            return literals[literal[1:-1]]
+            try:
+                return literals[literal[1:-1]]
+            except KeyError:
+                raise ValueError(
+                    f"Malformed quoted string in input: {literal}. "
+                    "Check for unclosed quotes or unmatched quote characters."
+                ) from None
         else:
             return literal
     return [unquote(item) for item in s.split(delimiter) if item.strip()]
```