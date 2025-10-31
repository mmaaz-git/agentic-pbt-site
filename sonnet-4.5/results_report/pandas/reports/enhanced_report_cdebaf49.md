# Bug Report: pandas.io.json._normalize.convert_to_line_delimits operator precedence causes data corruption

**Target**: `pandas.io.json._normalize.convert_to_line_delimits`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `convert_to_line_delimits` function has an operator precedence bug that causes it to incorrectly process non-array inputs, leading to data corruption by stripping characters from valid JSON objects and plain text strings.

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

if __name__ == "__main__":
    test_convert_to_line_delimits_only_processes_arrays()
```

<details>

<summary>
**Failing input**: `'00'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/53/hypo.py", line 15, in <module>
    test_convert_to_line_delimits_only_processes_arrays()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/53/hypo.py", line 5, in test_convert_to_line_delimits_only_processes_arrays
    @settings(max_examples=1000)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/53/hypo.py", line 12, in test_convert_to_line_delimits_only_processes_arrays
    assert result == s, f"Non-array input should be returned unchanged"
           ^^^^^^^^^^^
AssertionError: Non-array input should be returned unchanged
Falsifying example: test_convert_to_line_delimits_only_processes_arrays(
    s='00',  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
from pandas.io.json._normalize import convert_to_line_delimits

# Test case 1: Valid JSON array - should be processed
valid_array = '[1, 2, 3]'
result1 = convert_to_line_delimits(valid_array)
print(f"Input:  {repr(valid_array)}")
print(f"Output: {repr(result1)}")
print(f"Processed: {result1 != valid_array}")
print()

# Test case 2: Valid JSON object - should NOT be processed
valid_object = '{"a": 1}'
result2 = convert_to_line_delimits(valid_object)
print(f"Input:  {repr(valid_object)}")
print(f"Output: {repr(result2)}")
print(f"Unchanged: {result2 == valid_object}")
print()

# Test case 3: Malformed input (object start, array end) - should NOT be processed
malformed1 = '{"a": 1}]'
result3 = convert_to_line_delimits(malformed1)
print(f"Input:  {repr(malformed1)}")
print(f"Output: {repr(result3)}")
print(f"Unchanged: {result3 == malformed1}")
print()

# Test case 4: Malformed input (array start, object end) - should NOT be processed
malformed2 = '[1, 2, 3}'
result4 = convert_to_line_delimits(malformed2)
print(f"Input:  {repr(malformed2)}")
print(f"Output: {repr(result4)}")
print(f"Unchanged: {result4 == malformed2}")
print()

# Test case 5: Plain text - should NOT be processed
plain_text = 'plain text'
result5 = convert_to_line_delimits(plain_text)
print(f"Input:  {repr(plain_text)}")
print(f"Output: {repr(result5)}")
print(f"Unchanged: {result5 == plain_text}")
```

<details>

<summary>
Output showing incorrect processing of non-array inputs
</summary>
```
Input:  '[1, 2, 3]'
Output: '1\n 2\n 3\n'
Processed: True

Input:  '{"a": 1}'
Output: '"a": 1\n'
Unchanged: False

Input:  '{"a": 1}]'
Output: '{"a": 1}]'
Unchanged: True

Input:  '[1, 2, 3}'
Output: '1\n 2\n 3\n'
Unchanged: False

Input:  'plain text'
Output: 'lain tex\n'
Unchanged: False
```
</details>

## Why This Is A Bug

The function's docstring states it "converts JSON lists to line delimited JSON" and the code comment at line 36-37 explicitly says "Determine we have a JSON list to turn to lines otherwise just return the json object, only lists can".

However, the condition at line 38 has an operator precedence bug:

```python
if not s[0] == "[" and s[-1] == "]":
    return s
```

Due to Python's operator precedence (`not` binds tighter than `and`), this parses as:
```python
if (s[0] != "[") and (s[-1] == "]"):
    return s
```

This means the function only returns the input unchanged when:
- The string does NOT start with '['
- AND the string ends with ']'

This is the opposite of the intended behavior. The function should only process strings that both start with '[' AND end with ']' (valid JSON arrays).

As demonstrated in the reproduction script:
- **Valid JSON objects** like `'{"a": 1}'` are incorrectly processed, having their first and last characters stripped, resulting in `'"a": 1\n'` (data corruption)
- **Malformed JSON** like `'[1, 2, 3}'` (array start, object end) is incorrectly processed
- **Plain text** like `'plain text'` is incorrectly processed, becoming `'lain tex\n'` (first and last chars stripped)
- Only inputs matching the specific pattern `(not starting with '[') AND (ending with ']')` are left unchanged

## Relevant Context

The `convert_to_line_delimits` function is an internal helper in `pandas/io/json/_normalize.py`. While it's not part of the public API, it's used internally by pandas for JSON processing. The function uses `convert_json_to_lines` from `pandas._libs.writers` to do the actual line-delimited conversion.

The bug causes silent data corruption - it strips the first and last characters from any input that doesn't match its flawed condition, then passes the corrupted string to `convert_json_to_lines`. This can lead to invalid JSON being processed and potentially cause downstream errors or incorrect data analysis.

Documentation: https://github.com/pandas-dev/pandas/blob/main/pandas/io/json/_normalize.py#L32-L42

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