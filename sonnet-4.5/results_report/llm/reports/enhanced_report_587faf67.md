# Bug Report: llm.utils.truncate_string Violates Maximum Length Contract

**Target**: `llm.utils.truncate_string`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `truncate_string` function violates its documented contract by returning strings longer than the specified `max_length` parameter when `max_length < 3`, always returning at least 3 characters due to the hardcoded ellipsis "...".

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

# Run the test
test_truncate_string_max_length_property()
```

<details>

<summary>
**Failing input**: `text='0', max_length=0`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/54/hypo.py", line 14, in <module>
    test_truncate_string_max_length_property()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/54/hypo.py", line 5, in test_truncate_string_max_length_property
    @given(st.text(min_size=1), st.integers(min_value=0, max_value=100))
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/54/hypo.py", line 8, in test_truncate_string_max_length_property
    assert len(result) <= max_length, (
           ^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: truncate_string violated max_length constraint: len('...') = 3 > 0
Falsifying example: test_truncate_string_max_length_property(
    text='0',
    max_length=0,
)
```
</details>

## Reproducing the Bug

```python
from llm.utils import truncate_string

# Test case 1: max_length=1
print("Test case 1: max_length=1")
result = truncate_string("hello", max_length=1)
print(f"Result: {repr(result)}")
print(f"Length: {len(result)}")
print(f"Expected max length: 1")
print(f"Violation: {len(result) > 1}")
print()

# Test case 2: max_length=0
print("Test case 2: max_length=0")
result = truncate_string("hello", max_length=0)
print(f"Result: {repr(result)}")
print(f"Length: {len(result)}")
print(f"Expected max length: 0")
print(f"Violation: {len(result) > 0}")
print()

# Test case 3: max_length=2
print("Test case 3: max_length=2")
result = truncate_string("hello", max_length=2)
print(f"Result: {repr(result)}")
print(f"Length: {len(result)}")
print(f"Expected max length: 2")
print(f"Violation: {len(result) > 2}")
print()

# Test minimal failing case: text="ab", max_length=1
print("Test minimal case: text='ab', max_length=1")
result = truncate_string("ab", max_length=1)
print(f"Result: {repr(result)}")
print(f"Length: {len(result)}")
print(f"Expected max length: 1")
print(f"Violation: {len(result) > 1}")
print()

# Assertion test that will fail
print("Running assertion test...")
try:
    result = truncate_string("hello", max_length=1)
    assert len(result) <= 1, f"Length {len(result)} exceeds max_length of 1"
    print("Assertion passed")
except AssertionError as e:
    print(f"AssertionError: {e}")
```

<details>

<summary>
Output demonstrating contract violation
</summary>
```
Test case 1: max_length=1
Result: 'hel...'
Length: 6
Expected max length: 1
Violation: True

Test case 2: max_length=0
Result: 'he...'
Length: 5
Expected max length: 0
Violation: True

Test case 3: max_length=2
Result: 'hell...'
Length: 7
Expected max length: 2
Violation: True

Test minimal case: text='ab', max_length=1
Result: '...'
Length: 3
Expected max length: 1
Violation: True

Running assertion test...
AssertionError: Length 6 exceeds max_length of 1
```
</details>

## Why This Is A Bug

The function's docstring at `/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages/llm/utils.py:445-456` explicitly documents:
- **Function purpose**: "Truncate a string to a maximum length"
- **Parameter `max_length`**: "Maximum length of the result string"
- **No documented exceptions** for small max_length values

The word "maximum" has a clear mathematical meaning - an upper bound that should not be exceeded. When `max_length < 3`, the function executes line 476:
```python
return text[: max_length - 3] + "..."
```

This code unconditionally appends "..." (3 characters) to the truncated text. For `max_length < 3`:
- When `max_length=2`: Returns `text[-1:] + "..."` → minimum 4 characters (but actually more due to negative slicing behavior)
- When `max_length=1`: Returns `text[-2:] + "..."` → minimum 3 characters
- When `max_length=0`: Returns `text[-3:] + "..."` → minimum 3 characters

The function always returns at least 3 characters ("..."), violating the maximum length constraint for any `max_length < 3`.

## Relevant Context

This bug could cause real issues in production systems:

1. **Database constraints**: Fields with CHAR(1) or CHAR(2) constraints would reject the truncated values
2. **UI layouts**: Fixed-width display fields expecting single characters would overflow
3. **Network protocols**: Binary protocols with strict byte limits would fail
4. **Data validation**: Systems validating string length would reject the output

The function is part of the `llm` package (https://github.com/simonw/llm), a command-line tool for interacting with Large Language Models. The `truncate_string` utility is likely used throughout the codebase for formatting output and managing token limits.

## Proposed Fix

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