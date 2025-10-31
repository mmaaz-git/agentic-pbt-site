# Bug Report: llm.utils._parse_kwargs Confusing Errors on Unbalanced Brackets

**Target**: `llm.utils._parse_kwargs`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_parse_kwargs` function allows the bracket depth counter to become negative when parsing input with unbalanced closing brackets. This causes the parser to incorrectly group tokens, leading to confusing error messages that don't clearly indicate the root cause (unbalanced brackets).

## Property-Based Test

```python
from hypothesis import given, strategies as st
from llm.utils import _parse_kwargs

@given(st.text())
def test_parse_kwargs_balanced_brackets(s):
    if s.count('[') != s.count(']'):
        return
    if s.count('{') != s.count('}'):
        return
    if s.count('(') != s.count(')'):
        return
    try:
        _parse_kwargs(s)
    except ValueError as e:
        if 'unbalanced' in str(e).lower() or 'bracket' in str(e).lower():
            assert False, "Should not mention brackets for balanced input"
```

**Failing input**: `'key1=[1,2], key2=], key3=3'` (unbalanced closing bracket)

## Reproducing the Bug

```python
from llm.utils import _parse_kwargs

input_str = 'key1=[1,2], key2=], key3=3'
print(f"Input: {input_str}")

try:
    result = _parse_kwargs(input_str)
    print(f"Result: {result}")
except ValueError as e:
    print(f"Error: {e}")
```

**Output**:
```
Error: Value for 'key2' is not valid JSON: ], key3=3
```

The error message suggests that `'], key3=3'` is the value for `key2`, but the actual problem is that the parser didn't split on the comma after `key2=]` because the depth counter went negative (-1) when encountering the unbalanced `]`.

## Why This Is A Bug

1. **Incorrect token grouping**: The parser treats `key2=], key3=3` as a single token instead of splitting it at the comma, because `depth = -1` when it encounters the comma.

2. **Confusing error message**: The error says the JSON value `'], key3=3'` is invalid, which is technically true, but doesn't help the user understand that the real problem is an unbalanced bracket earlier in the input.

3. **No validation**: The function doesn't validate that brackets are balanced, allowing `depth` to become negative and causing downstream confusion.

Better behavior would be to detect the unbalanced bracket and raise an error like:
```
ValueError: Unbalanced brackets: unexpected ']' at position 20
```

## Fix

```diff
--- a/llm/utils.py
+++ b/llm/utils.py
@@ -556,6 +556,7 @@ def _parse_kwargs(arg_str: str) -> Dict[str, Any]:
     """Parse key=value pairs where each value is valid JSON."""
     tokens = []
     buf = []
     depth = 0
     in_string = False
     string_char = ""
@@ -580,6 +581,8 @@ def _parse_kwargs(arg_str: str) -> Dict[str, Any]:
                 buf.append(ch)
             elif ch in "}])":
                 depth -= 1
+                if depth < 0:
+                    raise ValueError(f"Unbalanced brackets: unexpected '{ch}'")
                 buf.append(ch)
             elif ch == "," and depth == 0:
                 tokens.append("".join(buf).strip())
@@ -589,6 +592,10 @@ def _parse_kwargs(arg_str: str) -> Dict[str, Any]:
     if buf:
         tokens.append("".join(buf).strip())

+    # Check for unclosed opening brackets
+    if depth > 0:
+        raise ValueError(f"Unbalanced brackets: {depth} unclosed opening bracket(s)")
+
     kwargs: Dict[str, Any] = {}
     for token in tokens:
         if not token:
```