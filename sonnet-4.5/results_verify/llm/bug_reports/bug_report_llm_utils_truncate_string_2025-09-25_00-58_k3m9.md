# Bug Report: llm.utils.truncate_string Length Constraint Violation

**Target**: `llm.utils.truncate_string`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `truncate_string` function violates its documented contract when `max_length` is less than 3, returning strings longer than the specified `max_length`.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from llm.utils import truncate_string

@given(
    st.text(),
    st.integers(min_value=1, max_value=1000),
    st.booleans(),
    st.booleans()
)
def test_truncate_string_length_constraint(text, max_length, normalize_whitespace, keep_end):
    result = truncate_string(text, max_length, normalize_whitespace, keep_end)
    assert len(result) <= max_length
```

**Failing input**: `text='00', max_length=1, normalize_whitespace=False, keep_end=False`

## Reproducing the Bug

```python
from llm.utils import truncate_string

result = truncate_string("00", max_length=1)
print(f"Result: '{result}'")
print(f"Length: {len(result)}")

result2 = truncate_string("hello", max_length=2)
print(f"Result: '{result2}'")
print(f"Length: {len(result2)}")
```

Output:
```
Result: '...'
Length: 3
Result: 'hell...'
Length: 7
```

## Why This Is A Bug

The function's docstring explicitly states:

> **Args:**
>   - max_length: Maximum length of the result string
>
> **Returns:**
>   Truncated string

When `max_length` is 1 or 2, the function returns "..." (3 characters) or longer strings, violating the documented constraint that the result should not exceed `max_length`.

The issue is in line 476 of `/llm/utils.py`:

```python
return text[: max_length - 3] + "..."
```

When `max_length` is 1, `text[:max_length - 3]` becomes `text[:-2]` which is not what we want, and adding "..." makes it 3 characters total.

## Fix

```diff
--- a/utils.py
+++ b/utils.py
@@ -473,7 +473,11 @@ def truncate_string(
         return text[:cutoff] + "... " + text[-cutoff:]
     else:
         # Fall back to simple truncation for very small max_length
-        return text[: max_length - 3] + "..."
+        if max_length < 3:
+            # For very small max_length, just truncate without ellipsis
+            return text[:max_length]
+        else:
+            return text[: max_length - 3] + "..."
```