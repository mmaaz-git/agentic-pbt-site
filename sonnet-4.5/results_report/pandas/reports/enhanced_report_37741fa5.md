# Bug Report: pandas.io.json._normalize.convert_to_line_delimits Operator Precedence Bug Causing Data Corruption

**Target**: `pandas.io.json._normalize.convert_to_line_delimits`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `convert_to_line_delimits` function has an operator precedence bug that causes it to incorrectly process and corrupt strings that start with `[` but don't end with `]`, violating its documented behavior of only converting valid JSON lists.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pandas.io.json._normalize import convert_to_line_delimits

@given(st.text(min_size=1).filter(lambda s: s[0] == '[' and s[-1] != ']'))
def test_convert_to_line_delimits_malformed_list(s):
    result = convert_to_line_delimits(s)
    assert result == s, f"Malformed JSON list {repr(s)} should be returned unchanged, got {repr(result)}"

if __name__ == "__main__":
    test_convert_to_line_delimits_malformed_list()
```

<details>

<summary>
**Failing input**: `'['`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/7/hypo.py", line 10, in <module>
    test_convert_to_line_delimits_malformed_list()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/7/hypo.py", line 5, in test_convert_to_line_delimits_malformed_list
    def test_convert_to_line_delimits_malformed_list(s):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/7/hypo.py", line 7, in test_convert_to_line_delimits_malformed_list
    assert result == s, f"Malformed JSON list {repr(s)} should be returned unchanged, got {repr(result)}"
           ^^^^^^^^^^^
AssertionError: Malformed JSON list '[' should be returned unchanged, got '\n'
Falsifying example: test_convert_to_line_delimits_malformed_list(
    s='[',
)
```
</details>

## Reproducing the Bug

```python
from pandas.io.json._normalize import convert_to_line_delimits

# Test case 1: String that starts with [ but doesn't end with ]
s1 = "[abc"
result1 = convert_to_line_delimits(s1)
print(f"Input:  {repr(s1)}")
print(f"Output: {repr(result1)}")
print(f"Expected: {repr(s1)} (unchanged)")
print()

# Test case 2: Just a single [ character
s2 = "["
result2 = convert_to_line_delimits(s2)
print(f"Input:  {repr(s2)}")
print(f"Output: {repr(result2)}")
print(f"Expected: {repr(s2)} (unchanged)")
print()

# Test case 3: Valid JSON list
s3 = "[1,2,3]"
result3 = convert_to_line_delimits(s3)
print(f"Input:  {repr(s3)}")
print(f"Output: {repr(result3)}")
print(f"Expected: Line-delimited output")
print()

# Test case 4: String ending with ] but not starting with [
s4 = "abc]"
result4 = convert_to_line_delimits(s4)
print(f"Input:  {repr(s4)}")
print(f"Output: {repr(result4)}")
print(f"Expected: {repr(s4)} (unchanged)")
```

<details>

<summary>
Output shows data corruption for malformed JSON inputs
</summary>
```
Input:  '[abc'
Output: 'ab\n'
Expected: '[abc' (unchanged)

Input:  '['
Output: '\n'
Expected: '[' (unchanged)

Input:  '[1,2,3]'
Output: '1\n2\n3\n'
Expected: Line-delimited output

Input:  'abc]'
Output: 'abc]'
Expected: 'abc]' (unchanged)
```
</details>

## Why This Is A Bug

The function's docstring explicitly states it "converts JSON lists to line delimited JSON". According to the JSON specification (RFC 7159), a valid JSON list/array must begin with `[` and end with `]`. The function should only process strings that satisfy both conditions, returning all other inputs unchanged.

However, the current implementation has an operator precedence error on line 38:

```python
if not s[0] == "[" and s[-1] == "]":
    return s
```

Due to Python's operator precedence rules, this expression evaluates as `(not s[0] == "[") and s[-1] == "]"` instead of the intended `not (s[0] == "[" and s[-1] == "]")`.

This causes the function to incorrectly process malformed inputs:
- For `s = "[abc"`: The condition evaluates to `(not True) and False` = `False and False` = `False`, so the function doesn't return early and incorrectly processes the string by stripping the first and last characters, resulting in `"ab"` being passed to `convert_json_to_lines`
- For `s = "["`: Similarly evaluates to `False`, causing the single bracket to be processed into just a newline
- For valid JSON lists like `"[1,2,3]"`: The condition correctly evaluates to `False`, allowing proper processing
- For `s = "abc]"`: The condition evaluates to `True and True` = `True`, correctly returning unchanged

The code comment on lines 36-37 reinforces the intended behavior: "Determine we have a JSON list to turn to lines otherwise just return the json object". The implementation contradicts this by processing non-JSON-list strings.

## Relevant Context

The `convert_to_line_delimits` function is used internally in pandas for JSON normalization operations. It relies on `convert_json_to_lines` from `pandas._libs.writers` which expects the input to have already had its enclosing brackets stripped. The bug causes data corruption by:

1. Incorrectly identifying malformed JSON as valid JSON lists
2. Stripping the first and last characters from these malformed strings
3. Passing the corrupted data to downstream processing

This is particularly problematic because:
- Data corruption occurs silently without error messages
- Users may not realize their data has been modified incorrectly
- The corruption can propagate through data pipelines

The function is located at: `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/io/json/_normalize.py:32-42`

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