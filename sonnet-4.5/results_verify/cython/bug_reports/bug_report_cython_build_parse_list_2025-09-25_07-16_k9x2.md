# Bug Report: Cython.Build.Dependencies.parse_list KeyError on Quoted Strings

**Target**: `Cython.Build.Dependencies.parse_list`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `parse_list()` function crashes with `KeyError` when parsing quoted strings due to a mismatch between the label format created by `strip_string_literals()` (with trailing underscore) and the label format expected by the internal `unquote()` function (without trailing underscore).

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from Cython.Build.Dependencies import parse_list


@settings(max_examples=1000)
@given(st.lists(st.text(min_size=0, max_size=20), min_size=0, max_size=10))
def test_parse_list_quoted_bracket_format_no_crash(items):
    s = '[' + ', '.join(f'"{item}"' for item in items) + ']'
    result = parse_list(s)
```

**Failing input**: `items=['']` (empty string in list)

## Reproducing the Bug

```python
from Cython.Build.Dependencies import parse_list

parse_list('[""]')
```

Output:
```
KeyError: ''
```

Additional failing cases:
```python
parse_list("'")
parse_list('["\\"]')
parse_list('[a, "", b]')
parse_list('"')
```

## Why This Is A Bug

The `parse_list()` function is a public API used to parse distutils configuration values from source code comments (e.g., `# distutils: libraries = foo bar`). It should handle any valid quoted string, including empty strings. The crash occurs due to two related issues:

1. **Empty string literals**: `strip_string_literals()` doesn't capture empty string literals (`""` or `''`) in its mapping, leaving them as-is in the normalized code. When `unquote()` tries to look up the empty key, it fails with `KeyError: ''`.

2. **Label format mismatch**: `strip_string_literals()` creates placeholder labels with format `__Pyx_L{N}_` (with trailing underscore). When `unquote()` processes a quoted placeholder like `"__Pyx_L1_"`, it uses `literal[1:-1]` which removes both the opening quote AND the final character, producing `__Pyx_L1` without the underscore, causing `KeyError: '__Pyx_L1'`.

## Fix

```diff
--- a/Cython/Build/Dependencies.py
+++ b/Cython/Build/Dependencies.py
@@ -128,8 +128,13 @@ def parse_list(s):
     s, literals = strip_string_literals(s)
     def unquote(literal):
         literal = literal.strip()
+        if not literal:
+            return literal
         if literal[0] in "'\"":
-            return literals[literal[1:-1]]
+            if literal == '""' or literal == "''":
+                return ''
+            key = literal[1:-1]
+            return literals.get(key, key)
         else:
             return literal
```

Alternative fix in `strip_string_literals()` to properly capture empty strings would be more robust but requires changes to the string parsing logic.