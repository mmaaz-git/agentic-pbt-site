# Bug Report: llm.utils.truncate_string Violates Maximum Length Contract for Small Values

**Target**: `llm.utils.truncate_string`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `truncate_string` function violates its documented contract when `max_length < 3`, returning strings that significantly exceed the specified maximum length instead of respecting the length limit.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from llm.utils import truncate_string

@given(st.text(), st.integers(min_value=1, max_value=1000))
def test_truncate_string_length(text, max_length):
    result = truncate_string(text, max_length)
    assert len(result) <= max_length, f"Result length {len(result)} exceeds max_length {max_length} for text='{text}'"

# Run the test
if __name__ == "__main__":
    test_truncate_string_length()
```

<details>

<summary>
**Failing input**: `text='00', max_length=1`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/23/hypo.py", line 11, in <module>
    test_truncate_string_length()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/23/hypo.py", line 5, in test_truncate_string_length
    def test_truncate_string_length(text, max_length):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/23/hypo.py", line 7, in test_truncate_string_length
    assert len(result) <= max_length, f"Result length {len(result)} exceeds max_length {max_length} for text='{text}'"
           ^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Result length 3 exceeds max_length 1 for text='00'
Falsifying example: test_truncate_string_length(
    text='00',
    max_length=1,
)
```
</details>

## Reproducing the Bug

```python
from llm.utils import truncate_string

# Test case that crashes - max_length=1
result = truncate_string("hello world", 1)
print(f"Result: '{result}'")
print(f"Length: {len(result)}")
print(f"Expected max: 1")
print()

# Additional failing cases
print("Additional failing cases:")
test_cases = [
    ("test", 1),
    ("test", 2),
    ("example", 1),
    ("example", 2),
    ("a", 1),
    ("ab", 2),
]

for text, max_len in test_cases:
    result = truncate_string(text, max_len)
    print(f"truncate_string('{text}', {max_len}) = '{result}' (length={len(result)})")
```

<details>

<summary>
Output demonstrates contract violation for max_length < 3
</summary>
```
Result: 'hello wor...'
Length: 12
Expected max: 1

Additional failing cases:
truncate_string('test', 1) = 'te...' (length=5)
truncate_string('test', 2) = 'tes...' (length=6)
truncate_string('example', 1) = 'examp...' (length=8)
truncate_string('example', 2) = 'exampl...' (length=9)
truncate_string('a', 1) = 'a' (length=1)
truncate_string('ab', 2) = 'ab' (length=2)
```
</details>

## Why This Is A Bug

The function's docstring explicitly states that `max_length` is the "Maximum length of the result string" (line 450 in `/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages/llm/utils.py`). This creates a clear contract that the returned string should never exceed `max_length` characters.

However, when `max_length < 3`, the function violates this contract due to incorrect handling in the else branch (line 476):
```python
return text[: max_length - 3] + "..."
```

When `max_length` is 1 or 2:
- `max_length=1` results in `text[:-2]` which uses negative indexing, taking all but the last 2 characters of the text, then appending "..." (3 chars), resulting in strings much longer than 1 character
- `max_length=2` results in `text[:-1]` which takes all but the last character, then appends "...", again violating the 2-character limit

The function correctly handles the case when the text is already shorter than `max_length` (lines 463-464), returning the text unchanged. However, for longer texts with `max_length < 3`, the negative slice indexing causes the bug.

## Relevant Context

The truncate_string function is part of the LLM utility module and is likely used for display purposes in command-line interfaces or logging. The function has special handling for `keep_end=True` mode (lines 469-473) where it correctly validates that `max_length >= 9` before attempting to show both start and end, but lacks similar validation for the default truncation mode.

The bug only manifests for edge cases where `max_length` is 1 or 2, which are uncommon in practice since truncating to such small lengths is rarely useful. However, the function should either handle these cases correctly or raise an appropriate error rather than silently violating its contract.

Code location: `/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages/llm/utils.py:439-477`

## Proposed Fix

```diff
--- a/llm/utils.py
+++ b/llm/utils.py
@@ -473,5 +473,8 @@ def truncate_string(
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