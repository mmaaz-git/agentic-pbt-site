# Bug Report: llm.utils.truncate_string Violates max_length Contract

**Target**: `llm.utils.truncate_string`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `truncate_string` function violates its contract by returning strings longer than the specified `max_length` when `max_length < 3`.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from llm.utils import truncate_string

@given(st.text(), st.integers(min_value=1, max_value=1000))
def test_truncate_string_length(text, max_length):
    result = truncate_string(text, max_length)
    assert len(result) <= max_length
```

**Failing input**: `text="hello world"`, `max_length=1`

## Reproducing the Bug

```python
from llm.utils import truncate_string

result = truncate_string("hello world", max_length=1)
print(f"Result: '{result}'")
print(f"Result length: {len(result)}")

assert len(result) <= 1
```

Output:
```
Result: 'hel...'
Result length: 6
AssertionError
```

The function returns a 6-character string when `max_length=1`. This also fails for `max_length=2`.

## Why This Is A Bug

The function's docstring and signature promise that the result will be truncated to `max_length`, which is a reasonable expectation. The fundamental property `len(truncate_string(text, max_length)) <= max_length` should always hold, but it's violated for `max_length < 3`.

The bug is in this branch:
```python
else:
    return text[: max_length - 3] + "..."
```

When `max_length < 3`, the expression `text[: max_length - 3]` becomes `text[: negative_number]`, which creates a slice from the beginning of the string backwards, leaving too many characters before appending "..." (3 chars).

## Fix

```diff
--- a/llm/utils.py
+++ b/llm/utils.py
@@ -472,7 +472,11 @@ def truncate_string(
         cutoff = (max_length - 5) // 2
         return text[:cutoff] + "... " + text[-cutoff:]
     else:
-        # Fall back to simple truncation for very small max_length
-        return text[: max_length - 3] + "..."
+        if max_length < 3:
+            # For very small max_length, just truncate without ellipsis
+            return text[:max_length]
+        else:
+            # Fall back to simple truncation
+            return text[: max_length - 3] + "..."
```