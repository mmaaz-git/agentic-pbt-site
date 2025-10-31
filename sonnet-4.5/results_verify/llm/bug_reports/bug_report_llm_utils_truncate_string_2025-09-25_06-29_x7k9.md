# Bug Report: llm.utils.truncate_string Violates max_length Constraint

**Target**: `llm.utils.truncate_string`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `truncate_string` function violates its documented contract by returning strings longer than `max_length` when `max_length < 3`.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from llm.utils import truncate_string


@given(st.text(min_size=5), st.integers(min_value=1, max_value=2))
def test_truncate_string_respects_max_length(text, max_length):
    result = truncate_string(text, max_length=max_length)
    assert len(result) <= max_length, \
        f"truncate_string({repr(text)}, {max_length}) returned {repr(result)} " \
        f"with length {len(result)} > {max_length}"
```

**Failing input**: `text="hello", max_length=1`

## Reproducing the Bug

```python
from llm.utils import truncate_string

result = truncate_string("hello", max_length=1)

print(f"Result: {repr(result)}")
print(f"Expected max length: 1")
print(f"Actual length: {len(result)}")

assert len(result) <= 1
```

Output:
```
Result: '...'
Expected max length: 1
Actual length: 3
AssertionError
```

## Why This Is A Bug

The function's docstring states: "Truncate a string to a maximum length" and the parameter is named `max_length`. This creates a clear expectation that the result will never exceed `max_length` characters. However, when `max_length < 3`, the function returns `"..."` (3 characters), violating this invariant.

Looking at the code (lines 474-476):
```python
else:
    # Fall back to simple truncation for very small max_length
    return text[: max_length - 3] + "..."
```

When `max_length = 1`:
- `text[:1 - 3]` = `text[:-2]` = (usually empty for short strings)
- Result: `"" + "..."` = `"..."` with length 3

This violates the contract that `len(result) <= max_length`.

## Fix

```diff
--- a/llm/utils.py
+++ b/llm/utils.py
@@ -471,6 +471,10 @@ def truncate_string(
         cutoff = (max_length - 5) // 2
         return text[:cutoff] + "... " + text[-cutoff:]
     else:
-        # Fall back to simple truncation for very small max_length
-        return text[: max_length - 3] + "..."
+        # For very small max_length, truncate without ellipsis if needed
+        if max_length < 3:
+            return text[:max_length]
+        else:
+            return text[: max_length - 3] + "..."
```

This ensures that when `max_length` is 1 or 2, the function returns the truncated text without appending `"..."`, thus respecting the `max_length` constraint.