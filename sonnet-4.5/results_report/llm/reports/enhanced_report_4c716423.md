# Bug Report: llm.utils.truncate_string Violates Max Length Contract

**Target**: `llm.utils.truncate_string()`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `truncate_string()` function violates its documented contract by returning strings longer than `max_length` when `max_length < 3`, returning ellipsis strings that exceed the specified maximum length parameter.

## Property-Based Test

```python
#!/usr/bin/env python3
"""Hypothesis property-based test for truncate_string"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings
from llm.utils import truncate_string

@given(st.text(min_size=1), st.integers(min_value=1, max_value=1000))
@settings(max_examples=1000)
def test_truncate_string_length_invariant(text, max_length):
    """Test that truncate_string respects max_length contract"""
    result = truncate_string(text, max_length=max_length)
    assert len(result) <= max_length, f"Result '{result}' (len={len(result)}) exceeds max_length={max_length}"

if __name__ == "__main__":
    test_truncate_string_length_invariant()
```

<details>

<summary>
**Failing input**: `text='00', max_length=1`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/55/hypo.py", line 18, in <module>
    test_truncate_string_length_invariant()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/55/hypo.py", line 11, in test_truncate_string_length_invariant
    @settings(max_examples=1000)
                   ^^^
  File "/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/55/hypo.py", line 15, in test_truncate_string_length_invariant
    assert len(result) <= max_length, f"Result '{result}' (len={len(result)}) exceeds max_length={max_length}"
           ^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Result '...' (len=3) exceeds max_length=1
Falsifying example: test_truncate_string_length_invariant(
    text='00',
    max_length=1,
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages/llm/utils.py:467
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""Minimal reproduction of truncate_string bug"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

from llm.utils import truncate_string

# Test cases that violate the max_length contract
test_cases = [
    ('00', 1),
    ('000', 2),
    ('0000', 3),
]

print("Demonstrating truncate_string violating max_length contract")
print("=" * 60)

for text, max_length in test_cases:
    result = truncate_string(text, max_length=max_length)
    print(f"\nInput: text='{text}', max_length={max_length}")
    print(f"Result: '{result}'")
    print(f"Result length: {len(result)}")
    print(f"Expected max length: {max_length}")
    print(f"Contract violated: {len(result) > max_length}")
```

<details>

<summary>
Contract violations demonstrated for max_length values 1 and 2
</summary>
```
Demonstrating truncate_string violating max_length contract
============================================================

Input: text='00', max_length=1
Result: '...'
Result length: 3
Expected max length: 1
Contract violated: True

Input: text='000', max_length=2
Result: '00...'
Result length: 5
Expected max length: 2
Contract violated: True

Input: text='0000', max_length=3
Result: '...'
Result length: 3
Expected max length: 3
Contract violated: False
```
</details>

## Why This Is A Bug

The function's docstring explicitly documents that the `max_length` parameter represents the "Maximum length of the result string" with no exceptions or minimum value requirements. The function violates this contract when `max_length < 3` and the input text exceeds `max_length`.

The issue occurs in the fallback truncation logic at line 476 of `/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages/llm/utils.py`:

```python
return text[: max_length - 3] + "..."
```

When `max_length=1`: The slice becomes `text[:-2]`, which for strings of length 2+ results in an empty string, yielding `"" + "..." = "..."` (length 3).

When `max_length=2`: The slice becomes `text[:-1]`, which returns all but the last character. For input `'000'`, this yields `"00" + "..." = "00..."` (length 5).

This violates the documented invariant that the result should never exceed `max_length` characters.

## Relevant Context

The `truncate_string` function is used extensively throughout the `llm` CLI tool for formatting output display:
- Line 1921-1922 in cli.py: Truncating prompts and responses
- Line 1931 in cli.py: Truncating fragment content
- Line 1998-2001 in cli.py: Truncating system prompts
- Line 2021 in cli.py: Truncating tool outputs
- Line 2859 in cli.py: Truncating result content

While the default `max_length` is 100, and typical usage likely involves larger values, the function should still honor its contract for all valid inputs. The violation could cause formatting issues in contexts where strict length limits are required, such as terminal display constraints or fixed-width output formatting.

Documentation link: The function's docstring at line 445-456 clearly states the contract without any minimum value restrictions.

## Proposed Fix

```diff
--- a/llm/utils.py
+++ b/llm/utils.py
@@ -473,7 +473,16 @@ def truncate_string(
         return text[:cutoff] + "... " + text[-cutoff:]
     else:
         # Fall back to simple truncation for very small max_length
-        return text[: max_length - 3] + "..."
+        if max_length < 3:
+            # Too short for ellipsis, just truncate
+            return text[:max_length]
+        else:
+            # Use ellipsis only when there's room
+            ellipsis_len = min(3, max_length)
+            if len(text) <= max_length:
+                return text
+            else:
+                return text[: max_length - ellipsis_len] + "." * ellipsis_len
```