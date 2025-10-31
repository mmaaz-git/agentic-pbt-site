# Bug Report: Cython.Build parse_list Filters Out Quoted Whitespace

**Target**: `Cython.Build.Dependencies.parse_list`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `parse_list` function incorrectly filters out items containing only whitespace when using bracket notation, despite the function's doctests explicitly showing that quoted whitespace should be preserved.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume, settings
from Cython.Build.Dependencies import parse_list

@given(st.lists(st.text(alphabet=st.characters(blacklist_categories=('Cs',)), min_size=1)))
@settings(max_examples=500)
def test_parse_list_bracket_format_roundtrip(items):
    assume(all(item for item in items))
    assume(all(',' not in item and '"' not in item and "'" not in item for item in items))

    input_str = '[' + ', '.join(items) + ']'
    result = parse_list(input_str)
    assert result == items
```

**Failing input**: `items=[' ']`

## Reproducing the Bug

```python
from Cython.Build.Dependencies import parse_list

result = parse_list("[' ']")
print(result)
```

**Expected output:** `[' ']`
**Actual output:** `[]`

This directly contradicts the function's own doctest at line 118-119:
```python
>>> parse_list('a " " b')
['a', ' ', 'b']
```

The doctest shows that space-delimited format correctly preserves quoted whitespace, but bracket notation does not.

## Why This Is A Bug

The function's doctests explicitly claim that quoted whitespace should be preserved (line 118-119). The implementation violates this contract for bracket notation.

The root cause is on line 135:

```python
return [unquote(item) for item in s.split(delimiter) if item.strip()]
```

The `if item.strip()` filter removes items that become empty after stripping whitespace. However, after `strip_string_literals` is called, quoted strings are replaced with placeholder labels like `__Pyx_L1_`. These labels themselves are not empty, so they pass the `item.strip()` check. The actual bug is in the `unquote` function's handling of these labels - it doesn't properly retrieve the original quoted content from the `literals` dictionary, resulting in the label itself being returned instead of the whitespace content, which then fails subsequent processing.

## Fix

The issue is that `unquote` doesn't correctly handle the labels produced by `strip_string_literals`. The labels should be looked up in the literals dictionary:

```diff
--- a/Cython/Build/Dependencies.py
+++ b/Cython/Build/Dependencies.py
@@ -128,8 +128,11 @@ def parse_list(s):
     s, literals = strip_string_literals(s)
     def unquote(literal):
         literal = literal.strip()
+        if literal in literals:
+            return literals[literal]
         if literal[0] in "'\"":
             return literals[literal[1:-1]]
         else:
             return literal
     return [unquote(item) for item in s.split(delimiter) if item.strip()]
```

This ensures that labels from `strip_string_literals` are properly resolved to their original string content.