# Bug Report: pandas.io.json._normalize.convert_to_line_delimits Operator Precedence Error

**Target**: `pandas.io.json._normalize.convert_to_line_delimits`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `convert_to_line_delimits` function has an operator precedence bug that causes it to incorrectly strip characters from non-array JSON strings, resulting in data corruption when the input is not a JSON array.

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

if __name__ == "__main__":
    test_convert_to_line_delimits_non_list_unchanged()
```

<details>

<summary>
**Failing input**: `'0'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/43/hypo.py", line 12, in <module>
    test_convert_to_line_delimits_non_list_unchanged()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/43/hypo.py", line 5, in test_convert_to_line_delimits_non_list_unchanged
    @settings(max_examples=100, deadline=None)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/43/hypo.py", line 9, in test_convert_to_line_delimits_non_list_unchanged
    assert result == json_str
           ^^^^^^^^^^^^^^^^^^
AssertionError
Falsifying example: test_convert_to_line_delimits_non_list_unchanged(
    json_str='0',
)
```
</details>

## Reproducing the Bug

```python
from pandas.io.json._normalize import convert_to_line_delimits

# Test with single digit
print(f"Input: '0' -> Output: {repr(convert_to_line_delimits('0'))}")

# Test with multi-digit number
print(f"Input: '123' -> Output: {repr(convert_to_line_delimits('123'))}")

# Test with empty JSON object
print(f"Input: '{{}}' -> Output: {repr(convert_to_line_delimits('{}'))}")

# Test with JSON string
print(f"Input: '\"hello\"' -> Output: {repr(convert_to_line_delimits('\"hello\"'))}")

# Test with actual JSON array (should be processed)
print(f"Input: '[1,2,3]' -> Output: {repr(convert_to_line_delimits('[1,2,3]'))}")

# Test with JSON object
print(f"Input: '{{\"a\":1}}' -> Output: {repr(convert_to_line_delimits('{\"a\":1}'))}")
```

<details>

<summary>
Data corruption occurs for non-array JSON inputs
</summary>
```
Input: '0' -> Output: '\n'
Input: '123' -> Output: '2\n'
Input: '{}' -> Output: '\n'
Input: '"hello"' -> Output: 'hello\n'
Input: '[1,2,3]' -> Output: '1\n2\n3\n'
Input: '{"a":1}' -> Output: '"a":1\n'
```
</details>

## Why This Is A Bug

The function's docstring states it "converts JSON lists to line delimited JSON" and the comment at line 36-37 explicitly says it should "just return the json object" if the input is not a JSON list. However, the function corrupts non-array JSON input by incorrectly stripping the first and last characters.

The bug occurs at line 38 in `/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/json/_normalize.py`:

```python
if not s[0] == "[" and s[-1] == "]":
    return s
```

Due to Python's operator precedence, this condition is evaluated as:
```python
if (not s[0] == "[") and (s[-1] == "]"):
    return s
```

This means the function only returns the input unchanged when:
1. The first character is NOT `[` (true for non-arrays)
2. **AND** the last character IS `]` (false for non-arrays)

Since non-array JSON strings don't end with `]`, the condition evaluates to False, causing the function to incorrectly proceed to lines 40-42 where it strips the first and last characters and passes the result through `convert_json_to_lines`.

For example, with input `'0'`:
- `not s[0] == "["` evaluates to `True` (since '0' != '[')
- `s[-1] == "]"` evaluates to `False` (since '0' != ']')
- `True and False` evaluates to `False`
- The function doesn't return early and proceeds to strip characters
- Line 40: `s = s[1:-1]` converts `'0'` to an empty string `''`
- Line 42: `convert_json_to_lines('')` returns `'\n'`

## Relevant Context

This function is used internally by pandas when exporting JSON with the `lines=True` parameter. While it's an internal function (not part of the public API), it's called by pandas' public JSON export functionality, meaning this bug can affect users who export non-array data to line-delimited JSON format.

The function is located at:
- File: `/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/json/_normalize.py`
- Lines: 32-42
- GitHub: https://github.com/pandas-dev/pandas/blob/main/pandas/io/json/_normalize.py

The bug represents a classic operator precedence error where the intended logic was to check if the string is NOT a JSON array (i.e., doesn't have both `[` at start AND `]` at end), but the actual implementation checks for an impossible condition.

## Proposed Fix

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