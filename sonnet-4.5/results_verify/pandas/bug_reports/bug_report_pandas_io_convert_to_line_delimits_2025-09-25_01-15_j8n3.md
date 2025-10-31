# Bug Report: pandas.io.json._normalize.convert_to_line_delimits Logic Error

**Target**: `pandas.io.json._normalize.convert_to_line_delimits`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `convert_to_line_delimits` function has a logic error in its condition that checks whether to process a JSON list. Due to operator precedence, the function incorrectly processes JSON objects, plain strings, and malformed JSON arrays.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pandas.io.json._normalize import convert_to_line_delimits


@given(s=st.text(min_size=2))
def test_convert_to_line_delimits_preserves_non_lists(s):
    if s[0] == "[" and s[-1] == "]":
        return

    result = convert_to_line_delimits(s)
    assert result == s
```

**Failing input**: `s='{"a": 1}'` (and many others)

## Reproducing the Bug

```python
from pandas.io.json._normalize import convert_to_line_delimits

json_object = '{"a": 1}'
result = convert_to_line_delimits(json_object)
print(f"Input:  {json_object!r}")
print(f"Output: {result!r}")
```

**Output:**
```
Input:  '{"a": 1}'
Output: '"a": 1\n'
```

The JSON object is incorrectly modified by having its first and last characters removed.

**Additional failing cases:**
- `'test'` → `'est\n'` (plain string modified)
- `'[test'` → `'test\n'` (malformed array modified)

## Why This Is A Bug

The function is documented to "Helper function that converts JSON lists to line delimited JSON." It should only process strings that are JSON arrays (start with `[` and end with `]`). However, the condition on line 37 is incorrect:

```python
if not s[0] == "[" and s[-1] == "]":
    return s
```

Due to operator precedence, this is parsed as:
```python
if (not (s[0] == "[")) and (s[-1] == "]"):
    return s
```

This means "return if first char is NOT `[` AND last char IS `]`". This is the **opposite** of the intended logic. The function should return when the string is NOT a JSON list (i.e., doesn't both start with `[` and end with `]`).

This causes the function to incorrectly process:
1. JSON objects like `'{"a": 1}'`
2. Plain strings like `'test'`
3. Malformed arrays like `'[test'`

All of these get their first and last characters removed, which corrupts the data.

## Fix

```diff
--- a/pandas/io/json/_normalize.py
+++ b/pandas/io/json/_normalize.py
@@ -34,7 +34,7 @@ def convert_to_line_delimits(s: str) -> str:
     """
     # Determine we have a JSON list to turn to lines otherwise just return the
     # json object, only lists can
-    if not s[0] == "[" and s[-1] == "]":
+    if not (s[0] == "[" and s[-1] == "]"):
         return s
     s = s[1:-1]
```