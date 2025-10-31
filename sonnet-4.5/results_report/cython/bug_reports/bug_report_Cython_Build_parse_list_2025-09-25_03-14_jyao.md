# Bug Report: Cython.Build.Dependencies.parse_list Empty String KeyError

**Target**: `Cython.Build.Dependencies.parse_list`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `parse_list` function crashes with a KeyError when parsing bracket-delimited lists containing empty strings (e.g., `[""]`).

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
from Cython.Build.Dependencies import parse_list

@given(st.lists(st.text()))
def test_parse_list_bracket_delimited_round_trip(items):
    assume(all('"' not in item for item in items))
    quoted_items = [f'"{item}"' for item in items]
    input_str = '[' + ', '.join(quoted_items) + ']'
    result = parse_list(input_str)
    assert result == items
```

**Failing input**: `items=['']`

## Reproducing the Bug

```python
from Cython.Build.Dependencies import parse_list

parse_list('[""]')
```

Output:
```
KeyError: ''
```

The function also fails with single-quoted empty strings: `parse_list("['']")` raises the same KeyError.

## Why This Is A Bug

The `parse_list` function is designed to parse distutils-style list configurations. According to its docstring examples, it should handle quoted strings in bracket-delimited lists. An empty string is a valid list element that users might legitimately include in configuration (e.g., `include_dirs=[""]`).

The root cause is a mismatch between `strip_string_literals` and `parse_list`:
- `strip_string_literals` intentionally ignores empty string literals (`""` or `''`) and leaves them unchanged (see line 78: `if quote and len(quote) != 2`)
- But `unquote` in `parse_list` assumes all quoted strings have been replaced with labels and tries to look them up in the `literals` dict

## Fix

```diff
--- a/Cython/Build/Dependencies.py
+++ b/Cython/Build/Dependencies.py
@@ -129,7 +129,10 @@ def parse_list(s):
     s, literals = strip_string_literals(s)
     def unquote(literal):
         literal = literal.strip()
-        if literal[0] in "'\""' and literal[0] == literal[-1]:
+        if len(literal) >= 2 and literal[0] in "'\""' and literal[0] == literal[-1]:
+            if len(literal) == 2:
+                # Empty string literal that wasn't replaced by strip_string_literals
+                return ''
             return literals[literal[1:-1]]
         else:
             return literal
```
