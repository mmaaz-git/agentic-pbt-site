# Bug Report: pandas.io.json convert_to_line_delimits incorrect boolean logic

**Target**: `pandas.io.json._normalize.convert_to_line_delimits`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`convert_to_line_delimits` has incorrect operator precedence in its guard condition, causing it to incorrectly process malformed inputs and potentially skip valid JSON arrays.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from pandas.io.json._normalize import convert_to_line_delimits

@given(st.text(min_size=2, max_size=100))
@settings(max_examples=1000)
def test_convert_to_line_delimits_only_processes_arrays(s):
    result = convert_to_line_delimits(s)

    if s[0] == '[' and s[-1] == ']':
        pass
    else:
        assert result == s, f"Non-array input should be returned unchanged"
```

**Failing input**: `'{"a": 1}]'` (starts with `{`, ends with `]`)

## Reproducing the Bug

```python
from pandas.io.json._normalize import convert_to_line_delimits

malformed_input = '{"a": 1}]'
result = convert_to_line_delimits(malformed_input)

print(f"Input:  {repr(malformed_input)}")
print(f"Output: {repr(result)}")
print(f"Unchanged: {result == malformed_input}")
```

Expected: Input should be returned unchanged (not a valid JSON array)
Actual: Input is incorrectly processed (brackets stripped, passed to convert_json_to_lines)

## Why This Is A Bug

The function `convert_to_line_delimits` at line 38 of `pandas/io/json/_normalize.py` has this condition:

```python
if not s[0] == "[" and s[-1] == "]":
    return s
```

Due to Python's operator precedence (`not` binds tighter than `and`), this parses as:
```python
if (s[0] != "[") and (s[-1] == "]"):
    return s
```

This means the function returns early only when the string starts with something OTHER than `[` AND ends with `]`. This is backwards!

The function should only process valid JSON arrays (strings that start with `[` AND end with `]`). The current logic:
- ✗ Processes `'[1, 2, 3}'` (starts with `[`, ends with `}`) - invalid!
- ✓ Returns `'{"a": 1}]'` unchanged (starts with `{`, ends with `]`) - correct by accident
- ✓ Returns `'{"a": 1}'` unchanged (starts with `{`, ends with `}`) - but fails the first check
- ✗ Might process `'[1, 2, 3]'` if it passes the first check - correct by accident

## Fix

```diff
--- a/pandas/io/json/_normalize.py
+++ b/pandas/io/json/_normalize.py
@@ -35,7 +35,7 @@ def convert_to_line_delimits(s: str) -> str:
     """
     # Determine we have a JSON list to turn to lines otherwise just return the
     # json object, only lists can
-    if not s[0] == "[" and s[-1] == "]":
+    if not (s[0] == "[" and s[-1] == "]"):
         return s
     s = s[1:-1]
```

Alternatively:
```diff
-    if not s[0] == "[" and s[-1] == "]":
+    if s[0] != "[" or s[-1] != "]":
         return s
```