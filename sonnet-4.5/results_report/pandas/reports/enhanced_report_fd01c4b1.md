# Bug Report: pandas.io.json._normalize.convert_to_line_delimits Operator Precedence Error

**Target**: `pandas.io.json._normalize.convert_to_line_delimits`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `convert_to_line_delimits` function contains an operator precedence bug that causes it to incorrectly process non-array JSON strings, resulting in data corruption where strings like '00' become '\n' and JSON objects like '{"foo": "bar"}' lose their outer braces.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pandas.io.json._normalize as normalize


@given(st.text(min_size=2))
def test_convert_to_line_delimits_property(json_str):
    result = normalize.convert_to_line_delimits(json_str)

    if json_str[0] == "[" and json_str[-1] == "]":
        # This is a JSON array, should be converted to line-delimited format
        pass  # We don't validate the exact format here
    else:
        # Non-array strings should be returned unchanged
        assert result == json_str, f"Non-list string should be unchanged: {json_str!r} -> {result!r}"


# Run the test
if __name__ == "__main__":
    test_convert_to_line_delimits_property()
```

<details>

<summary>
**Failing input**: `'00'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/34/hypo.py", line 19, in <module>
    test_convert_to_line_delimits_property()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/34/hypo.py", line 6, in test_convert_to_line_delimits_property
    def test_convert_to_line_delimits_property(json_str):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/34/hypo.py", line 14, in test_convert_to_line_delimits_property
    assert result == json_str, f"Non-list string should be unchanged: {json_str!r} -> {result!r}"
           ^^^^^^^^^^^^^^^^^^
AssertionError: Non-list string should be unchanged: '00' -> '\n'
Falsifying example: test_convert_to_line_delimits_property(
    json_str='00',  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
import pandas.io.json._normalize as normalize

# Test cases to demonstrate the bug
test_cases = [
    ("00", "00"),  # Non-array string should be unchanged
    ('{"foo": "bar"}', '{"foo": "bar"}'),  # JSON object should be unchanged
    ("x]", "x]"),  # String ending with ] but not starting with [ should be unchanged
    ("[1, 2, 3]", "Line-delimited format"),  # JSON array should be converted
]

print("Testing convert_to_line_delimits function:")
print("=" * 50)

for input_str, expected_desc in test_cases:
    result = normalize.convert_to_line_delimits(input_str)

    if expected_desc == "Line-delimited format":
        # For arrays, we expect line-delimited output
        print(f"Input: {input_str!r}")
        print(f"Output: {result!r}")
        print(f"Expected: Line-delimited JSON format")
    else:
        # For non-arrays, output should equal input
        if result != input_str:
            print(f"BUG FOUND!")
            print(f"  Input:    {input_str!r}")
            print(f"  Output:   {result!r}")
            print(f"  Expected: {expected_desc!r}")
        else:
            print(f"OK: {input_str!r} -> {result!r}")
    print("-" * 50)
```

<details>

<summary>
Data corruption: '00' becomes '\n', JSON objects lose outer braces
</summary>
```
Testing convert_to_line_delimits function:
==================================================
BUG FOUND!
  Input:    '00'
  Output:   '\n'
  Expected: '00'
--------------------------------------------------
BUG FOUND!
  Input:    '{"foo": "bar"}'
  Output:   '"foo": "bar"\n'
  Expected: '{"foo": "bar"}'
--------------------------------------------------
OK: 'x]' -> 'x]'
--------------------------------------------------
Input: '[1, 2, 3]'
Output: '1\n 2\n 3\n'
Expected: Line-delimited JSON format
--------------------------------------------------
```
</details>

## Why This Is A Bug

The function's docstring states "Helper function that converts JSON lists to line delimited JSON." The code comment on lines 36-37 explicitly says "Determine we have a JSON list to turn to lines otherwise just return the json object, only lists can". This clearly indicates the function should:

1. Check if the input is a JSON array (starts with '[' AND ends with ']')
2. If it IS an array, convert it to line-delimited format
3. If it is NOT an array, return it unchanged

However, the conditional on line 38 has an operator precedence error:
```python
if not s[0] == "[" and s[-1] == "]":
    return s
```

Due to Python's operator precedence where `not` binds tighter than `and`, this is evaluated as:
```python
if (not s[0] == "[") and (s[-1] == "]"):
    return s
```

This means the function only returns the string unchanged when:
- First character is NOT '[' AND last character IS ']'

For all other inputs (including '00', '{"foo": "bar"}'), the function incorrectly continues to line 40 where it strips the first and last characters with `s = s[1:-1]` and passes the result to `convert_json_to_lines`. This causes:
- '00' → '0' → convert_json_to_lines('0') → '\n'
- '{"foo": "bar"}' → '"foo": "bar"' → convert_json_to_lines('"foo": "bar"') → '"foo": "bar"\n'

This violates the documented behavior and corrupts non-array JSON data.

## Relevant Context

The `convert_to_line_delimits` function is located in `/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/json/_normalize.py` at line 32. It's used internally by pandas for JSON normalization operations. Line-delimited JSON (also known as NDJSON or JSON Lines) is a format where each line contains a separate valid JSON value, commonly used for streaming large datasets.

The bug affects any code path that uses this function to process JSON strings, particularly when the input is not a JSON array but gets incorrectly processed as one. Since this is in a private module (_normalize), it's primarily used internally by pandas, but the data corruption it causes can propagate through the JSON processing pipeline.

Documentation link: https://pandas.pydata.org/docs/reference/api/pandas.json_normalize.html

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