# Bug Report: llm.utils.truncate_string Returns Strings Exceeding max_length for Small Values

**Target**: `llm.utils.truncate_string`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `truncate_string` function violates its documented contract by returning strings longer than the specified `max_length` parameter when `max_length` is less than 3, always returning "..." (3 characters) regardless of the constraint.

## Property-Based Test

```python
import sys
sys.path.insert(0, "/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages")

from hypothesis import given, strategies as st
from llm.utils import truncate_string

@given(st.text(), st.integers(min_value=1, max_value=1000))
def test_truncate_string_length_bound(text, max_length):
    result = truncate_string(text, max_length=max_length)
    assert len(result) <= max_length, f"Result '{result}' (len={len(result)}) exceeds max_length={max_length}"

# Run the property-based test
test_truncate_string_length_bound()
```

<details>

<summary>
**Failing input**: `text='00', max_length=1`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/7/hypo.py", line 13, in <module>
    test_truncate_string_length_bound()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/7/hypo.py", line 8, in test_truncate_string_length_bound
    def test_truncate_string_length_bound(text, max_length):
                   ^^^
  File "/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/7/hypo.py", line 10, in test_truncate_string_length_bound
    assert len(result) <= max_length, f"Result '{result}' (len={len(result)}) exceeds max_length={max_length}"
           ^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Result '...' (len=3) exceeds max_length=1
Falsifying example: test_truncate_string_length_bound(
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
import sys
sys.path.insert(0, "/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages")

from llm.utils import truncate_string

# Test the specific failing case from the bug report
result = truncate_string('00', max_length=1)
print(f"Input: text='00', max_length=1")
print(f"Expected: length <= 1")
print(f"Result: '{result}'")
print(f"Actual length: {len(result)}")
print(f"Violates contract: {len(result) > 1}")
print()

# Test a few more edge cases
test_cases = [
    ('abc', 2),
    ('hello', 3),
    ('test', 0),
    ('', 1),
    ('x', 1),
    ('xy', 1),
]

print("Additional test cases:")
for text, max_length in test_cases:
    result = truncate_string(text, max_length=max_length)
    violation = len(result) > max_length
    print(f"  text='{text}', max_length={max_length} -> '{result}' (len={len(result)}) {'VIOLATES' if violation else 'OK'}")
```

<details>

<summary>
Contract violation: Returns '...' (length 3) when max_length=1
</summary>
```
Input: text='00', max_length=1
Expected: length <= 1
Result: '...'
Actual length: 3
Violates contract: True

Additional test cases:
  text='abc', max_length=2 -> 'ab...' (len=5) VIOLATES
  text='hello', max_length=3 -> '...' (len=3) OK
  text='test', max_length=0 -> 't...' (len=4) VIOLATES
  text='', max_length=1 -> '' (len=0) OK
  text='x', max_length=1 -> 'x' (len=1) OK
  text='xy', max_length=1 -> '...' (len=3) VIOLATES
```
</details>

## Why This Is A Bug

The function's docstring explicitly states that `max_length` is the "Maximum length of the result string" with no caveats or exceptions. This creates a clear contract that the returned string's length should never exceed `max_length`. However, the implementation violates this contract in several ways:

1. **For max_length < 3**: The function always returns "..." (3 characters), exceeding the specified maximum
2. **Negative array slicing**: Line 476 computes `text[:max_length - 3] + "..."`, which for max_length=1 becomes `text[:-2]`, resulting in unexpected behavior
3. **Even worse for max_length=2**: Returns partial text plus "..." (e.g., 'ab...' for text='abc'), resulting in strings of length 5 or more

The code comment on line 475 says "Fall back to simple truncation for very small max_length" but the implementation doesn't actually handle these small values correctly. The function blindly appends "..." without checking if this would exceed the maximum length constraint.

## Relevant Context

This bug occurs in the `llm` package's utility module at `/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages/llm/utils.py`. The function is used for truncating strings in various contexts throughout the llm command-line tool.

The bug only manifests when:
- `max_length < 3` AND
- `len(text) > max_length` (triggering the truncation path)

When the text is already shorter than `max_length` (line 463) or when the text is empty (line 458), the function works correctly. The issue is specific to the truncation logic on line 476.

While this is an edge case that rarely occurs in practice (most users would use reasonable `max_length` values of 10+ for meaningful truncation), it still represents a clear violation of the documented API contract.

## Proposed Fix

```diff
--- a/llm/utils.py
+++ b/llm/utils.py
@@ -473,5 +473,8 @@ def truncate_string(
         return text[:cutoff] + "... " + text[-cutoff:]
     else:
         # Fall back to simple truncation for very small max_length
+        if max_length < 3:
+            # Can't fit ellipsis, just truncate without it
+            return text[:max_length]
         return text[: max_length - 3] + "..."
```