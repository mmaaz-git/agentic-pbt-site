# Bug Report: llm.utils.truncate_string Max Length Violation

**Target**: `llm.utils.truncate_string`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `truncate_string` function violates its documented contract by returning strings longer than the specified `max_length` parameter when `max_length < 3`.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st
from llm.utils import truncate_string

@settings(max_examples=500)
@given(st.text(min_size=1), st.integers(min_value=0, max_value=100))
def test_truncate_string_max_length_property(text, max_length):
    result = truncate_string(text, max_length=max_length)
    assert len(result) <= max_length, (
        f"truncate_string violated max_length constraint: "
        f"len({repr(result)}) = {len(result)} > {max_length}"
    )
```

**Failing input**: `text="ab"`, `max_length=1`

## Reproducing the Bug

```python
from llm.utils import truncate_string

result = truncate_string("hello", max_length=1)
print(f"Result: {repr(result)}")
print(f"Length: {len(result)}")

assert len(result) <= 1
```

**Output:**
```
Result: '...'
Length: 3
AssertionError
```

Additional failing examples:
- `truncate_string("hello", max_length=0)` returns `"..."` (length 3 > 0)
- `truncate_string("hello", max_length=2)` returns `"..."` (length 3 > 2)

## Why This Is A Bug

The function's docstring explicitly states it will "Truncate a string to a maximum length" and the parameter is documented as "Maximum length of the result string". This creates a clear contract that `len(result) <= max_length` must always hold.

When `max_length < 3`, the function executes line 476:
```python
return text[: max_length - 3] + "..."
```

Since the ellipsis `"..."` is always 3 characters long, the result will always have length >= 3, regardless of the specified `max_length`.

This violates the documented behavior and could cause issues for callers who rely on the max_length guarantee for UI layout, database fields, or other length-constrained contexts.

## Fix

```diff
--- a/llm/utils.py
+++ b/llm/utils.py
@@ -473,7 +473,10 @@ def truncate_string(
         return text[:cutoff] + "... " + text[-cutoff:]
     else:
         # Fall back to simple truncation for very small max_length
-        return text[: max_length - 3] + "..."
+        if max_length < 3:
+            return text[:max_length]
+        else:
+            return text[: max_length - 3] + "..."
```

This fix ensures that when `max_length < 3`, the function returns a simple truncation without the ellipsis, maintaining the length constraint while still providing reasonable behavior for edge cases.