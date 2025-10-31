# Bug Report: pandas.io.json._normalize.convert_to_line_delimits Operator Precedence Error Corrupts JSON Objects

**Target**: `pandas.io.json._normalize.convert_to_line_delimits`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `convert_to_line_delimits` function contains an operator precedence bug that causes it to incorrectly process non-list JSON strings, stripping the opening and closing braces from valid JSON objects and producing invalid, unparseable JSON output.

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

    assert json_obj == result, f"Non-list JSON should be unchanged. Input: {json_obj}, Output: {result}"

if __name__ == "__main__":
    test_convert_to_line_delimits_preserves_non_lists()
```

<details>

<summary>
**Failing input**: `{"0": 0}`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/43/hypo.py", line 15, in <module>
    test_convert_to_line_delimits_preserves_non_lists()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/43/hypo.py", line 6, in test_convert_to_line_delimits_preserves_non_lists
    def test_convert_to_line_delimits_preserves_non_lists(d):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/43/hypo.py", line 12, in test_convert_to_line_delimits_preserves_non_lists
    assert json_obj == result, f"Non-list JSON should be unchanged. Input: {json_obj}, Output: {result}"
           ^^^^^^^^^^^^^^^^^^
AssertionError: Non-list JSON should be unchanged. Input: {"0": 0}, Output: "0": 0

Falsifying example: test_convert_to_line_delimits_preserves_non_lists(
    d={'0': 0},  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
from pandas.io.json._normalize import convert_to_line_delimits

# Test with a simple JSON object
json_obj = '{"key": "value"}'
result = convert_to_line_delimits(json_obj)

print(f"Input:  {repr(json_obj)}")
print(f"Output: {repr(result)}")
print()

# Test with another JSON object
json_obj2 = '{"0": 0}'
result2 = convert_to_line_delimits(json_obj2)

print(f"Input:  {repr(json_obj2)}")
print(f"Output: {repr(result2)}")
print()

# Show the issue: the function should return non-lists unchanged
assert json_obj == result, f"Expected unchanged output for JSON object, but got corrupted JSON"
```

<details>

<summary>
AssertionError: Expected unchanged output for JSON object, but got corrupted JSON
</summary>
```
Input:  '{"key": "value"}'
Output: '"key": "value"\n'

Input:  '{"0": 0}'
Output: '"0": 0\n'

Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/43/repo.py", line 20, in <module>
    assert json_obj == result, f"Expected unchanged output for JSON object, but got corrupted JSON"
           ^^^^^^^^^^^^^^^^^^
AssertionError: Expected unchanged output for JSON object, but got corrupted JSON
```
</details>

## Why This Is A Bug

The function violates its documented behavior through an operator precedence error that causes data corruption.

### Documentation Contract
The function's docstring states: "Helper function that converts JSON lists to line delimited JSON."
An inline comment further clarifies: "Determine we have a JSON list to turn to lines otherwise just return the json object, only lists can"

This explicitly states:
1. JSON lists (arrays) should be converted to line-delimited JSON format
2. Non-lists should be returned unchanged ("just return the json object")
3. Only lists can be converted to line-delimited format

### The Operator Precedence Bug
The bug is in line 38 of `/pandas/io/json/_normalize.py`:

```python
if not s[0] == "[" and s[-1] == "]":
    return s
```

Due to Python's operator precedence, this evaluates as:
```python
if (not (s[0] == "[")) and (s[-1] == "]"):
```

This condition returns early ONLY when:
- The string does NOT start with "[" (not s[0] == "[" → True)
- AND the string DOES end with "]" (s[-1] == "]" → True)

For valid JSON objects like `{"key": "value"}`:
- `not (s[0] == "[")` evaluates to True (starts with "{", not "[")
- `s[-1] == "]"` evaluates to False (ends with "}", not "]")
- `True and False` = False
- The function does NOT return early and incorrectly processes the JSON object

### Data Corruption Process
When the function fails to return early for JSON objects:
1. It strips the first and last characters (line 40: `s = s[1:-1]`)
2. For `{"key": "value"}`, this becomes `"key": "value"` (missing braces)
3. It passes the corrupted string to `convert_json_to_lines()` which adds newlines
4. The output `"key": "value"\n` is invalid JSON that cannot be parsed

## Relevant Context

This function is used internally by pandas when calling `DataFrame.to_json(lines=True)` or similar JSON export operations. The JSON Lines format (newline-delimited JSON) is a standard format where each line contains a valid JSON value, commonly used for streaming and log processing.

The function correctly handles its intended use case (JSON arrays):
- Input: `[{"a": 1}, {"b": 2}]`
- Output: `{"a": 1}\n {"b": 2}\n`

However, when data serializes to a JSON object instead of an array, the function corrupts the output instead of returning it unchanged as documented. This affects users who call `to_json(lines=True)` on data that doesn't produce a JSON array.

Code location: `/pandas/io/json/_normalize.py:32-42`
Used by: `pandas.io.json._json.to_json()` when `lines=True` parameter is set

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