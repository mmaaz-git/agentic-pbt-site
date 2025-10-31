# Bug Report: llm.utils.truncate_string Violates max_length Contract for Small Values

**Target**: `llm.utils.truncate_string`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `truncate_string` function violates its documented contract by returning strings longer than `max_length` when `max_length < 3`, due to incorrect negative index slicing that results in nearly the full string plus ellipsis being returned.

## Property-Based Test

```python
#!/usr/bin/env python3
"""Hypothesis test for truncate_string bug"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
from llm.utils import truncate_string

@given(st.text(min_size=1), st.integers(min_value=1, max_value=1000))
def test_truncate_string_length_invariant(text, max_length):
    result = truncate_string(text, max_length)
    assert len(result) <= max_length, f"Failed: len({repr(result)}) = {len(result)} > {max_length}"

if __name__ == "__main__":
    test_truncate_string_length_invariant()
```

<details>

<summary>
**Failing input**: `text='00', max_length=1`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/61/hypo.py", line 16, in <module>
    test_truncate_string_length_invariant()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/61/hypo.py", line 11, in test_truncate_string_length_invariant
    def test_truncate_string_length_invariant(text, max_length):
                   ^^^
  File "/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/61/hypo.py", line 13, in test_truncate_string_length_invariant
    assert len(result) <= max_length, f"Failed: len({repr(result)}) = {len(result)} > {max_length}"
           ^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Failed: len('...') = 3 > 1
Falsifying example: test_truncate_string_length_invariant(
    text='00',
    max_length=1,
)
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""Minimal test case demonstrating truncate_string bug"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

from llm.utils import truncate_string

# Test case 1: max_length = 1
print("Test 1: truncate_string('hello world', 1)")
result = truncate_string("hello world", 1)
print(f"  Result: {repr(result)}")
print(f"  Length: {len(result)}")
print(f"  Expected max length: 1")
print(f"  Violation: {len(result) > 1}")
print()

# Test case 2: max_length = 2
print("Test 2: truncate_string('hello world', 2)")
result = truncate_string("hello world", 2)
print(f"  Result: {repr(result)}")
print(f"  Length: {len(result)}")
print(f"  Expected max length: 2")
print(f"  Violation: {len(result) > 2}")
print()

# Test case 3: max_length = 3 (should work correctly)
print("Test 3: truncate_string('hello world', 3)")
result = truncate_string("hello world", 3)
print(f"  Result: {repr(result)}")
print(f"  Length: {len(result)}")
print(f"  Expected max length: 3")
print(f"  Violation: {len(result) > 3}")
print()

# Test case 4: Short string with small max_length
print("Test 4: truncate_string('ab', 1)")
result = truncate_string("ab", 1)
print(f"  Result: {repr(result)}")
print(f"  Length: {len(result)}")
print(f"  Expected max length: 1")
print(f"  Violation: {len(result) > 1}")
```

<details>

<summary>
Output showing severe length violations
</summary>
```
Test 1: truncate_string('hello world', 1)
  Result: 'hello wor...'
  Length: 12
  Expected max length: 1
  Violation: True

Test 2: truncate_string('hello world', 2)
  Result: 'hello worl...'
  Length: 13
  Expected max length: 2
  Violation: True

Test 3: truncate_string('hello world', 3)
  Result: '...'
  Length: 3
  Expected max length: 3
  Violation: False

Test 4: truncate_string('ab', 1)
  Result: '...'
  Length: 3
  Expected max length: 1
  Violation: True

```
</details>

## Why This Is A Bug

The function's docstring states it will "Truncate a string to a maximum length" with parameter `max_length` described as "Maximum length of the result string". This creates an unambiguous contract that `len(result) <= max_length` should always hold.

The bug occurs in the else branch (lines 474-476 of llm/utils.py) when handling strings longer than `max_length`:
```python
else:
    # Fall back to simple truncation for very small max_length
    return text[: max_length - 3] + "..."
```

When `max_length < 3`, the expression `max_length - 3` becomes negative. In Python, negative indices in slicing mean "from the end", so:
- `text[:1-3]` becomes `text[:-2]` which returns all but the last 2 characters
- `text[:2-3]` becomes `text[:-1]` which returns all but the last 1 character

This causes the function to return nearly the entire original string plus "...", resulting in strings **much longer** than the specified `max_length`. For example, `truncate_string("hello world", 1)` returns `'hello wor...'` with length 12 instead of the maximum 1.

## Relevant Context

The function correctly handles the case where the input text is already shorter than or equal to `max_length` (line 463-464), returning the unmodified text. The bug only manifests when the text needs truncation AND `max_length < 3`.

The function also has a `keep_end` parameter that uses a different code path (lines 469-473) with a minimum length of 9, suggesting the developer was aware of edge cases with small `max_length` values but didn't properly handle them in the simple truncation path.

Function location: `/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages/llm/utils.py:439-476`

## Proposed Fix

```diff
--- a/llm/utils.py
+++ b/llm/utils.py
@@ -472,5 +472,9 @@ def truncate_string(
         cutoff = (max_length - 5) // 2
         return text[:cutoff] + "... " + text[-cutoff:]
     else:
-        # Fall back to simple truncation for very small max_length
-        return text[: max_length - 3] + "..."
+        # Fall back to simple truncation
+        if max_length < 3:
+            # For very small max_length, just truncate without ellipsis
+            return text[:max_length]
+        else:
+            return text[: max_length - 3] + "..."
```