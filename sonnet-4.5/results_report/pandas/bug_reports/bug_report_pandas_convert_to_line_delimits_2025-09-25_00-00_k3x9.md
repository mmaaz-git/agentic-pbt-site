# Bug Report: pandas.io.json._normalize.convert_to_line_delimits Operator Precedence

**Target**: `pandas.io.json._normalize.convert_to_line_delimits`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `convert_to_line_delimits` function has an operator precedence bug that causes it to incorrectly modify non-array JSON strings. The bug occurs when the input string does not start with `[` but also does not end with `]`.

## Property-Based Test

```python
from pandas.io.json._normalize import convert_to_line_delimits
from hypothesis import given, strategies as st, assume, settings

@given(st.text(alphabet=st.characters(min_codepoint=32, max_codepoint=126), min_size=1, max_size=100))
@settings(max_examples=100, deadline=None)
def test_convert_to_line_delimits_non_list_unchanged(json_str):
    assume(not (json_str.startswith('[') and json_str.endswith(']')))
    result = convert_to_line_delimits(json_str)
    assert result == json_str
```

**Failing input**: `'0'`

## Reproducing the Bug

```python
from pandas.io.json._normalize import convert_to_line_delimits

print(convert_to_line_delimits('0'))

print(convert_to_line_delimits('123'))

print(convert_to_line_delimits('{}'))

print(convert_to_line_delimits('"hello"'))
```

**Output:**
```
\n
2
{
hello
```

**Expected output:** The input strings should be returned unchanged since they are not JSON arrays.

## Why This Is A Bug

The function is documented to "convert JSON lists to line delimited JSON" and return non-list JSON unchanged. However, due to incorrect operator precedence in the condition on line 38, the function incorrectly strips the first and last characters from non-array JSON strings that don't end with `]`.

The buggy line:
```python
if not s[0] == "[" and s[-1] == "]":
    return s
```

is evaluated as:
```python
if (not s[0] == "[") and (s[-1] == "]"):
    return s
```

This means the function returns early only when the first character is NOT `[` AND the last character IS `]`, which is backwards. For input `'0'`:
- First character is NOT `[` → True
- Last character is NOT `]` → False
- True AND False → False → does not return early
- Proceeds to strip characters → produces `'\n'`

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

Alternatively, for better readability:
```diff
--- a/pandas/io/json/_normalize.py
+++ b/pandas/io/json/_normalize.py
@@ -35,7 +35,7 @@ def convert_to_line_delimits(s: str) -> str:
     """
     # Determine we have a JSON list to turn to lines otherwise just return the
     # json object, only lists can
-    if not s[0] == "[" and s[-1] == "]":
+    if s[0] != "[" or s[-1] != "]":
         return s
     s = s[1:-1]
```