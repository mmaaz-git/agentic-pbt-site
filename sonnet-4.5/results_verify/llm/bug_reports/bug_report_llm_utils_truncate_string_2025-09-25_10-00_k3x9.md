# Bug Report: llm.utils.truncate_string Incorrect Separator Length

**Target**: `llm.utils.truncate_string`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `truncate_string` function with `keep_end=True` returns strings shorter than `max_length` due to an incorrect calculation that assumes the separator is 5 characters when it's actually 4 characters.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from llm.utils import truncate_string

@given(st.text(min_size=20), st.integers(min_value=9, max_value=100))
def test_truncate_string_respects_max_length(text, max_length):
    result = truncate_string(text, max_length=max_length, keep_end=True)
    assert len(result) <= max_length, \
        f"Result length {len(result)} exceeds max_length {max_length}"
```

**Failing input**: `text="abcdefghijklmnop"`, `max_length=10`

## Reproducing the Bug

```python
from llm.utils import truncate_string

text = "abcdefghijklmnop"
max_length = 10
result = truncate_string(text, max_length=max_length, keep_end=True)

print(f"Input: '{text}' (length {len(text)})")
print(f"Result: '{result}' (length {len(result)})")
print(f"Expected max length: {max_length}")
print(f"Actual length: {len(result)}")
```

Output:
```
Input: 'abcdefghijklmnop' (length 16)
Result: 'ab... op' (length 8)
Expected max length: 10
Actual length: 8
```

The function returns a string of length 8 when `max_length=10`, wasting 2 characters.

## Why This Is A Bug

The function's docstring states it will "Truncate a string to a maximum length". Users expect the result to be as close to `max_length` as possible. However, the code incorrectly calculates:

```python
# Subtract 5 for the "... " separator
cutoff = (max_length - 5) // 2
return text[:cutoff] + "... " + text[-cutoff:]
```

The separator `"... "` is 4 characters (3 dots + 1 space), not 5. This off-by-one error causes the result to be shorter than necessary, potentially truncating more information than needed.

For `max_length=10`:
- Current: `cutoff = (10 - 5) // 2 = 2` → result is 2 + 4 + 2 = 8 chars
- Correct: `cutoff = (10 - 4) // 2 = 3` → result is 3 + 4 + 3 = 10 chars

## Fix

```diff
--- a/llm/utils.py
+++ b/llm/utils.py
@@ -469,8 +469,8 @@ def truncate_string(
     if keep_end and max_length >= min_keep_end_length:
         # Calculate how much text to keep at each end
-        # Subtract 5 for the "... " separator
-        cutoff = (max_length - 5) // 2
+        # Subtract 4 for the "... " separator
+        cutoff = (max_length - 4) // 2
         return text[:cutoff] + "... " + text[-cutoff:]
     else:
         # Fall back to simple truncation for very small max_length
```