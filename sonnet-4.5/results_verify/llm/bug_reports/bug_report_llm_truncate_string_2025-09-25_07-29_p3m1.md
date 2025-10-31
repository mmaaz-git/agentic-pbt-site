# Bug Report: llm.utils.truncate_string - Violates Length Invariant for Small max_length

**Target**: `llm.utils.truncate_string`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `truncate_string` function violates its documented invariant `len(result) <= max_length` when `max_length` is 1 or 2, returning "..." (3 characters) which exceeds the specified maximum.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from llm.utils import truncate_string

@given(st.text(min_size=1), st.integers(min_value=1, max_value=1000))
def test_truncate_string_length_invariant(text, max_length):
    result = truncate_string(text, max_length)
    assert len(result) <= max_length
```

**Failing input**: `text="hello world", max_length=1`

## Reproducing the Bug

```python
from llm.utils import truncate_string

result = truncate_string("hello world", 1)
print(f"Result: {repr(result)}, Length: {len(result)}")

result = truncate_string("hello world", 2)
print(f"Result: {repr(result)}, Length: {len(result)}")
```

**Output:**
```
Result: '...', Length: 3
Result: '...', Length: 3
```

## Why This Is A Bug

The function's docstring states it will "Truncate a string to a maximum length", creating an expectation that `len(result) <= max_length` always holds. When `max_length < 3`, the function returns "..." which has length 3, violating this invariant. While small values like 1 or 2 may be rare, they are valid inputs and the function should handle them correctly.

## Fix

```diff
--- a/llm/utils.py
+++ b/llm/utils.py
@@ -473,7 +473,11 @@ def truncate_string(
         return text[:cutoff] + "... " + text[-cutoff:]
     else:
-        # Fall back to simple truncation for very small max_length
-        return text[: max_length - 3] + "..."
+        if max_length < 3:
+            return text[:max_length]
+        else:
+            return text[: max_length - 3] + "..."
```