# Bug Report: llm.utils.truncate_string Returns Strings Exceeding max_length Parameter

**Target**: `llm.utils.truncate_string`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `truncate_string` function returns strings that exceed the specified `max_length` parameter when `max_length < 3`, directly violating its documented contract that promises to limit output to "Maximum length of the result string".

## Property-Based Test

```python
from hypothesis import given, strategies as st
import llm.utils

@given(
    st.text(min_size=1, max_size=1000),
    st.integers(min_value=1, max_value=500)
)
def test_truncate_string_length_property(text, max_length):
    result = llm.utils.truncate_string(text, max_length=max_length)
    assert len(result) <= max_length, f"len('{result}') = {len(result)} > max_length={max_length}"

if __name__ == "__main__":
    test_truncate_string_length_property()
```

<details>

<summary>
**Failing input**: `text='00', max_length=1`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/8/hypo.py", line 13, in <module>
    test_truncate_string_length_property()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/8/hypo.py", line 5, in test_truncate_string_length_property
    st.text(min_size=1, max_size=1000),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/8/hypo.py", line 10, in test_truncate_string_length_property
    assert len(result) <= max_length, f"len('{result}') = {len(result)} > max_length={max_length}"
           ^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: len('...') = 3 > max_length=1
Falsifying example: test_truncate_string_length_property(
    text='00',
    max_length=1,
)
```
</details>

## Reproducing the Bug

```python
import llm.utils

# Test case that demonstrates the bug
text = "Hello, World!"
max_length = 1

print(f"Input text: '{text}'")
print(f"Input text length: {len(text)}")
print(f"max_length: {max_length}")
print()

result = llm.utils.truncate_string(text, max_length=max_length)

print(f"Output: '{result}'")
print(f"Output length: {len(result)}")
print()
print(f"CONSTRAINT VIOLATION: len(result) = {len(result)} > max_length = {max_length}")
print(f"The function was asked to limit output to {max_length} character(s) but returned {len(result)} characters")
```

<details>

<summary>
Output demonstrates 14-character string returned when max_length=1
</summary>
```
Input text: 'Hello, World!'
Input text length: 13
max_length: 1

Output: 'Hello, Worl...'
Output length: 14

CONSTRAINT VIOLATION: len(result) = 14 > max_length = 1
The function was asked to limit output to 1 character(s) but returned 14 characters
```
</details>

## Why This Is A Bug

The function violates its fundamental contract in three ways:

1. **Parameter semantics**: The parameter is named `max_length`, which universally means "maximum" or "not to exceed" in programming contexts.

2. **Documentation promise**: The docstring explicitly states `max_length` is the "Maximum length of the result string" (line 450 of `/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages/llm/utils.py`).

3. **Magnitude of violation**: The function can return strings drastically longer than requested - 14 characters when asked for 1, a 1300% violation.

The bug occurs because the code at line 476 blindly subtracts 3 from `max_length` and appends "...":
```python
return text[: max_length - 3] + "..."
```

When `max_length < 3`, this creates a negative slice index. Python interprets `text[:-2]` as "all characters except the last 2", so for "Hello, World!" we get "Hello, Worl" + "..." = "Hello, Worl..." (14 characters).

## Relevant Context

The bug manifests in different ways depending on the input:
- `max_length=1` with 13-char input → returns 14 characters (1300% over limit)
- `max_length=2` with 13-char input → returns 15 characters (650% over limit)
- `max_length=1` with 2-char input → returns 3 characters (200% over limit)

This could cause production issues:
- Database field length constraint violations
- Buffer overflows in fixed-size UI elements
- Incorrect pagination calculations
- Data corruption when concatenating strings with assumed lengths

The function correctly handles the edge case for the `keep_end` parameter (checking for minimum length of 9), but fails to apply similar validation for simple truncation.

Code location: `/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages/llm/utils.py` lines 440-477

## Proposed Fix

```diff
--- a/llm/utils.py
+++ b/llm/utils.py
@@ -473,5 +473,9 @@ def truncate_string(
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