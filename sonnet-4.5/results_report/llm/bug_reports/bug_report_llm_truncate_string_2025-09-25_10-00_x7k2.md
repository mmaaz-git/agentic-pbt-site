# Bug Report: llm.utils.truncate_string Length Constraint Violation

**Target**: `llm.utils.truncate_string`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `truncate_string` function violates its fundamental contract when `max_length < 3`: it returns strings longer than the specified `max_length`, directly contradicting the parameter's documented purpose.

## Property-Based Test

```python
from hypothesis import given, strategies as st

@given(
    st.text(min_size=1, max_size=1000),
    st.integers(min_value=1, max_value=500)
)
def test_truncate_string_length_property(text, max_length):
    result = llm.utils.truncate_string(text, max_length=max_length)
    assert len(result) <= max_length, f"len('{result}') = {len(result)} > max_length={max_length}"
```

**Failing input**: `text = "Hello, World!"`, `max_length = 1`

## Reproducing the Bug

```python
import llm.utils

text = "Hello, World!"
max_length = 1
result = llm.utils.truncate_string(text, max_length=max_length)

print(f"Input: '{text}' (len={len(text)})")
print(f"max_length: {max_length}")
print(f"Output: '{result}' (len={len(result)})")
print(f"Constraint violated: {len(result)} > {max_length}")
```

**Output**:
```
Input: 'Hello, World!' (len=13)
max_length: 1
Output: 'Hello, Worl...' (len=14)
Constraint violated: 14 > 1
```

**Additional failing cases**:
- `max_length=1` → returns string of length 14 (for 13-char input)
- `max_length=2` → returns string of length 15 (for 13-char input)

## Why This Is A Bug

The function parameter is named `max_length` and the docstring states "Truncate a string to a maximum length". This creates a clear contract: the returned string should never exceed `max_length` characters.

However, the current implementation blindly appends "..." (3 characters) without checking if this would violate the constraint:

```python
return text[: max_length - 3] + "..."
```

When `max_length < 3`, the slice `text[: max_length - 3]` becomes `text[: negative_number]`, which in Python slices from the beginning up to that many characters from the end. This results in keeping most of the string, then adding "...", producing output much longer than `max_length`.

## Fix

```diff
--- a/llm/utils.py
+++ b/llm/utils.py
@@ -465,8 +465,14 @@ def truncate_string(

     # Minimum sensible length for keep_end is 9 characters: "a... z"
     min_keep_end_length = 9

     if keep_end and max_length >= min_keep_end_length:
         # Calculate how much text to keep at each end
         # Subtract 5 for the "... " separator
         cutoff = (max_length - 5) // 2
         return text[:cutoff] + "... " + text[-cutoff:]
     else:
-        # Fall back to simple truncation for very small max_length
-        return text[: max_length - 3] + "..."
+        # Fall back to simple truncation
+        if max_length < 3:
+            return text[:max_length]
+        else:
+            return text[: max_length - 3] + "..."
```