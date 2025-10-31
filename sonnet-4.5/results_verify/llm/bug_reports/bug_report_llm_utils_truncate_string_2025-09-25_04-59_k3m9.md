# Bug Report: llm.utils.truncate_string Length Invariant Violation

**Target**: `llm.utils.truncate_string`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `truncate_string` function violates its documented length invariant when `max_length < 3`, returning strings that exceed the specified maximum length instead of truncating them.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from llm.utils import truncate_string

@given(
    st.text(min_size=1),
    st.integers(min_value=1, max_value=1000),
    st.booleans(),
    st.booleans(),
)
def test_truncate_string_length_invariant(text, max_length, normalize_whitespace, keep_end):
    result = truncate_string(text, max_length, normalize_whitespace, keep_end)
    assert len(result) <= max_length, \
        f"Length invariant violated: len({repr(result)}) = {len(result)} > {max_length}"
```

**Failing input**: `text="hello", max_length=2, normalize_whitespace=False, keep_end=False`

## Reproducing the Bug

```python
from llm.utils import truncate_string

text = "hello"
max_length = 2
result = truncate_string(text, max_length)

print(f"Input: text={repr(text)}, max_length={max_length}")
print(f"Output: {repr(result)}")
print(f"Length: {len(result)}")

assert len(result) <= max_length
```

**Output:**
```
Input: text='hello', max_length=2
Output: 'hell...'
Length: 7
AssertionError: Length invariant violated
```

## Why This Is A Bug

The function's docstring states: "Truncate a string to a maximum length", and the parameter is named `max_length`, creating a clear contract that the returned string should never exceed this length. When `max_length < 3`, the function returns strings significantly longer than the maximum.

The bug occurs at line 476 in `utils.py`:
```python
return text[: max_length - 3] + "..."
```

When `max_length < 3`, the expression `max_length - 3` becomes negative, causing Python's negative slice indexing to activate. For example:
- `max_length = 2`: `text[:-1]` returns all but the last character
- `max_length = 1`: `text[:-2]` returns all but the last 2 characters
- `max_length = 0`: `text[:-3]` returns all but the last 3 characters

This creates strings much longer than intended when combined with the `"..."` suffix.

## Fix

```diff
--- a/llm/utils.py
+++ b/llm/utils.py
@@ -473,7 +473,11 @@ def truncate_string(
         return text[:cutoff] + "... " + text[-cutoff:]
     else:
         # Fall back to simple truncation for very small max_length
-        return text[: max_length - 3] + "..."
+        if max_length < 3:
+            # For very small max_length, just return the first characters
+            return text[:max_length]
+        else:
+            return text[: max_length - 3] + "..."
```

Alternatively, handle the edge case by ensuring `max_length >= 3` or adjusting the ellipsis:

```diff
--- a/llm/utils.py
+++ b/llm/utils.py
@@ -473,7 +473,9 @@ def truncate_string(
         return text[:cutoff] + "... " + text[-cutoff:]
     else:
         # Fall back to simple truncation for very small max_length
-        return text[: max_length - 3] + "..."
+        cutoff = max(0, max_length - 3)
+        suffix = "..." if max_length >= 3 else ""
+        return text[:cutoff] + suffix
```