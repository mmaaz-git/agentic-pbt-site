# Bug Report: pandas.io.json.convert_to_line_delimits Logic Error

**Target**: `pandas.io.json._normalize.convert_to_line_delimits`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `convert_to_line_delimits` function has a logic error due to incorrect operator precedence in its conditional check. The function incorrectly processes non-JSON-array strings by stripping their first and last characters, leading to data corruption.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from pandas.io.json._normalize import convert_to_line_delimits


@settings(max_examples=500)
@given(st.text(min_size=2))
def test_convert_to_line_delimits_only_processes_json_arrays(s):
    result = convert_to_line_delimits(s)
    is_json_array_format = s[0] == '[' and s[-1] == ']'

    if not is_json_array_format:
        assert result == s, (
            f"Non-JSON-array string should be returned unchanged. "
            f"Input: {s!r}, Output: {result!r}"
        )
```

**Failing inputs**:
- `"abc"` → Returns `"b"` instead of `"abc"`
- `"[abc"` → Attempts to process as JSON array instead of returning unchanged
- Any string that doesn't start with `'['` or doesn't end with `']'` (except strings ending with `']'` that don't start with `'['`)

## Reproducing the Bug

```python
from pandas.io.json._normalize import convert_to_line_delimits

result1 = convert_to_line_delimits("abc")
print(f"Input: 'abc', Output: {result1!r}")

result2 = convert_to_line_delimits("[invalid")
print(f"Input: '[invalid', Output: {result2!r}")
```

Expected behavior: Both inputs should be returned unchanged since they are not JSON arrays.
Actual behavior: Both inputs are processed, resulting in corrupted data.

## Why This Is A Bug

The conditional check at line 38 in `_normalize.py` has incorrect operator precedence:

```python
if not s[0] == "[" and s[-1] == "]":
    return s
```

Due to Python's operator precedence (`==` binds tighter than `not`, which binds tighter than `and`), this evaluates as:

```python
if (not (s[0] == "[")) and (s[-1] == "]"):
    return s
```

This returns the string unchanged **only when** it doesn't start with `'['` **and** it ends with `']'`.

For all other cases (including strings like `"abc"`, `"[abc"`, etc.), the function proceeds to strip the first and last characters and process them as if they were JSON arrays, which is incorrect.

The intended logic is to return the string unchanged unless it's a JSON array (starts with `'['` **and** ends with `']'`).

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

Alternative fix using De Morgan's law:

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