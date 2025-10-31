# Bug Report: Cython.Build.Dependencies.parse_list Empty String Crash

**Target**: `Cython.Build.Dependencies.parse_list`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `parse_list` function crashes with a `KeyError` when given a bracketed list containing an empty quoted string, such as `'[""]'`.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from Cython.Build.Dependencies import parse_list


@given(st.lists(st.text(min_size=0, max_size=10), min_size=0, max_size=20))
@settings(max_examples=1000)
def test_parse_list_no_crash_on_empty_strings(items):
    quoted_items = [f'"{item}"' for item in items]
    input_str = '[' + ', '.join(quoted_items) + ']'
    result = parse_list(input_str)
    assert isinstance(result, list)
```

**Failing input**: `items=['']`

## Reproducing the Bug

```python
from Cython.Build.Dependencies import parse_list

result = parse_list('[""]')
```

**Output**:
```
KeyError: ''
```

## Why This Is A Bug

The function's docstring includes examples showing it should handle quoted strings in bracketed lists:
```python
>>> parse_list('[a, ",a", "a,", ",", ]')
['a', ',a', 'a,', ',']
```

Empty quoted strings like `""` are valid input and should return `['']` (a list with one empty string), but instead the function crashes.

The root cause is that `strip_string_literals` doesn't extract empty strings into the literals map (it leaves them as `""` in the normalized code), but the `unquote` helper function assumes all quoted strings are in the map and tries to look them up with `literals[literal[1:-1]]`, causing a `KeyError` for empty strings.

## Fix

```diff
--- a/Cython/Build/Dependencies.py
+++ b/Cython/Build/Dependencies.py
@@ -129,7 +129,10 @@ def parse_list(s):
     s, literals = strip_string_literals(s)
     def unquote(literal):
         literal = literal.strip()
-        if literal[0] in "'\"":
+        if literal and literal[0] in "'\"":
+            key = literal[1:-1]
+            if not key:
+                return ''
             return literals[literal[1:-1]]
         else:
             return literal
```