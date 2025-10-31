# Bug Report: pandas.io.json.convert_to_line_delimits Operator Precedence Error Causes Data Corruption

**Target**: `pandas.io.json._normalize.convert_to_line_delimits`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `convert_to_line_delimits` function contains an operator precedence error that causes it to incorrectly process non-JSON-array strings, stripping their first and last characters and corrupting the data instead of returning them unchanged as intended.

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


if __name__ == "__main__":
    test_convert_to_line_delimits_only_processes_json_arrays()
```

<details>

<summary>
**Failing input**: `'00'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/39/hypo.py", line 19, in <module>
    test_convert_to_line_delimits_only_processes_json_arrays()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/39/hypo.py", line 6, in test_convert_to_line_delimits_only_processes_json_arrays
    @given(st.text(min_size=2))
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/39/hypo.py", line 12, in test_convert_to_line_delimits_only_processes_json_arrays
    assert result == s, (
           ^^^^^^^^^^^
AssertionError: Non-JSON-array string should be returned unchanged. Input: '00', Output: '\n'
Falsifying example: test_convert_to_line_delimits_only_processes_json_arrays(
    s='00',  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
from pandas.io.json._normalize import convert_to_line_delimits

# Test case 1: Regular string that is not a JSON array
result1 = convert_to_line_delimits("abc")
print(f"Input: 'abc'")
print(f"Expected output: 'abc'")
print(f"Actual output: {result1!r}")
print()

# Test case 2: String starting with [ but not ending with ]
result2 = convert_to_line_delimits("[invalid")
print(f"Input: '[invalid'")
print(f"Expected output: '[invalid'")
print(f"Actual output: {result2!r}")
print()

# Test case 3: String ending with ] but not starting with [
result3 = convert_to_line_delimits("test]")
print(f"Input: 'test]'")
print(f"Expected output: 'test]'")
print(f"Actual output: {result3!r}")
print()

# Test case 4: Valid JSON array format
result4 = convert_to_line_delimits("[valid]")
print(f"Input: '[valid]'")
print(f"Expected output: Should be processed (line-delimited)")
print(f"Actual output: {result4!r}")
```

<details>

<summary>
Data corruption occurs for non-JSON-array strings
</summary>
```
Input: 'abc'
Expected output: 'abc'
Actual output: 'b\n'

Input: '[invalid'
Expected output: '[invalid'
Actual output: 'invali\n'

Input: 'test]'
Expected output: 'test]'
Actual output: 'test]'

Input: '[valid]'
Expected output: Should be processed (line-delimited)
Actual output: 'valid\n'
```
</details>

## Why This Is A Bug

The function contains a critical operator precedence error at line 38 of `/pandas/io/json/_normalize.py`. The inline comment explicitly states: "Determine we have a JSON list to turn to lines otherwise just return the json object", indicating that non-JSON-array strings should be returned unchanged.

However, the conditional check:
```python
if not s[0] == "[" and s[-1] == "]":
    return s
```

Due to Python's operator precedence (where `==` binds tighter than `not`, which binds tighter than `and`), this evaluates as:
```python
if (not (s[0] == "[")) and (s[-1] == "]"):
    return s
```

This means the string is only returned unchanged when it **doesn't** start with `'['` **AND** it **does** end with `']'`. All other strings, including:
- Normal strings like `"abc"`, `"00"`, `"{}"`
- Partial JSON-like strings like `"[invalid"`
- Even valid JSON arrays like `"[valid]"`

...proceed to have their first and last characters stripped and get processed as if they were JSON arrays, resulting in data corruption. For example, `"abc"` becomes `"b\n"` and `"00"` becomes `"\n"`.

This violates the documented behavior in the code comment and causes silent data loss, which is particularly dangerous for an internal function that may be called from multiple places in the pandas codebase.

## Relevant Context

The `convert_to_line_delimits` function is located in the internal `pandas.io.json._normalize` module and is used for converting JSON arrays to line-delimited JSON format. It calls `convert_json_to_lines` from `pandas._libs.writers` after stripping the brackets.

The function's docstring is minimal ("Helper function that converts JSON lists to line delimited JSON") but the inline comment clearly indicates the intended behavior. The bug only affects strings that don't match the specific pattern of "not starting with '[' and ending with ']'", which coincidentally makes strings like `"test]"` work correctly, but for the wrong reason.

This is an internal/private function (indicated by the underscore in the module name), so it's not part of pandas' public API. However, data corruption in internal functions can lead to difficult-to-debug issues throughout the library.

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