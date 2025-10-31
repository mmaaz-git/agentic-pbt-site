# Bug Report: pandas.io.json convert_to_line_delimits Logic Error

**Target**: `pandas.io.json._normalize.convert_to_line_delimits`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `convert_to_line_delimits` function has an operator precedence bug that causes it to incorrectly process non-list JSON strings, corrupting valid JSON objects instead of preserving them unchanged.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
import json
from pandas.io.json._normalize import convert_to_line_delimits

@given(st.dictionaries(st.text(min_size=1, max_size=10), st.integers()))
def test_convert_to_line_delimits_preserves_non_lists(d):
    assume(len(d) > 0)

    json_obj = json.dumps(d)
    result = convert_to_line_delimits(json_obj)

    assert json_obj == result, "Non-list JSON should be unchanged"
```

**Failing input**: `{"0": 0}`

## Reproducing the Bug

```python
from pandas.io.json._normalize import convert_to_line_delimits

json_obj = '{"key": "value"}'
result = convert_to_line_delimits(json_obj)

print(f"Input:  {repr(json_obj)}")
print(f"Output: {repr(result)}")

assert json_obj == result
```

**Output:**
```
Input:  '{"key": "value"}'
Output: '"key": "value"\n'
AssertionError
```

The function corrupts the JSON object by stripping the first and last characters and passing it through `convert_json_to_lines`, even though the input is not a JSON list.

## Why This Is A Bug

The function's docstring and comment state it should "convert JSON lists to line delimited JSON" and "only lists can" be converted. JSON lists have the format `[...]`. For non-list JSON (like objects `{...}` or primitives), the function should return the input unchanged.

The bug is in line 38 of `_normalize.py`:

```python
if not s[0] == "[" and s[-1] == "]":
    return s
```

Due to operator precedence, this evaluates as:
```python
if (not (s[0] == "[")) and (s[-1] == "]"):
```

This returns early only when the string does NOT start with `[` AND DOES end with `]`, which matches malformed JSON like `{"a":1]` but NOT valid JSON objects like `{"a":1}`.

The intended logic is to return early when the input is NOT a list (i.e., NOT both starting with `[` AND ending with `]`).

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

The fix adds parentheses to ensure the condition checks "if NOT (is a list)" rather than "if (not starts with '[') AND (ends with ']')".