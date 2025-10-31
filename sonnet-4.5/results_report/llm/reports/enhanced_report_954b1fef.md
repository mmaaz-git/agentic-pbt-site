# Bug Report: llm.utils.truncate_string Returns Strings Longer Than max_length

**Target**: `llm.utils.truncate_string`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `truncate_string` function violates its documented contract by returning strings longer than `max_length` when `max_length < 3`.

## Property-Based Test

```python
from llm.utils import truncate_string
from hypothesis import given, strategies as st

@given(
    st.text(min_size=1, max_size=100),
    st.integers(min_value=0, max_value=10)
)
def test_truncate_string_length_invariant(text, max_length):
    result = truncate_string(text, max_length=max_length)
    assert len(result) <= max_length, \
        f"Result length {len(result)} exceeds max_length {max_length}: '{result}'"

# Run the test
if __name__ == "__main__":
    test_truncate_string_length_invariant()
```

<details>

<summary>
**Failing input**: `text='0', max_length=0`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/51/hypo.py", line 15, in <module>
    test_truncate_string_length_invariant()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/51/hypo.py", line 5, in test_truncate_string_length_invariant
    st.text(min_size=1, max_size=100),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/51/hypo.py", line 10, in test_truncate_string_length_invariant
    assert len(result) <= max_length, \
           ^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Result length 3 exceeds max_length 0: '...'
Falsifying example: test_truncate_string_length_invariant(
    text='0',
    max_length=0,
)
```
</details>

## Reproducing the Bug

```python
from llm.utils import truncate_string

text = "Hello world"

for max_length in [0, 1, 2, 3, 4]:
    result = truncate_string(text, max_length=max_length)
    print(f"max_length={max_length}: '{result}' (actual length={len(result)})")
```

<details>

<summary>
Output shows strings exceeding max_length for values < 3
</summary>
```
max_length=0: 'Hello wo...' (actual length=11)
max_length=1: 'Hello wor...' (actual length=12)
max_length=2: 'Hello worl...' (actual length=13)
max_length=3: '...' (actual length=3)
max_length=4: 'H...' (actual length=4)
```
</details>

## Why This Is A Bug

The function's docstring explicitly states that `max_length` is the "Maximum length of the result string" with no exceptions or caveats. However, when `max_length < 3`, the function returns strings significantly longer than the specified maximum, directly violating this documented contract.

The root cause is in the simple truncation logic at line 476 of `/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages/llm/utils.py`:

```python
return text[: max_length - 3] + "..."
```

When `max_length < 3`, the expression `max_length - 3` becomes negative (e.g., for `max_length=2`, it becomes `-1`). In Python, negative slice indices count from the end of the string, so `text[:-1]` returns everything except the last character, not an empty string as intended. This causes the function to return nearly the entire original text plus the ellipsis, making the result much longer than `max_length`.

## Relevant Context

The `truncate_string` function is part of the llm CLI tool's utility module and is likely used throughout the application to ensure text fits within terminal width constraints or API limits. The function accepts any non-negative integer for `max_length` without validation, implying all values should be handled correctly.

The bug manifests in these specific ways:
- `max_length=0`: Returns the original text minus 3 characters plus "..." (e.g., "Hello wo..." for "Hello world")
- `max_length=1`: Returns the original text minus 2 characters plus "..." (e.g., "Hello wor..." for "Hello world")
- `max_length=2`: Returns the original text minus 1 character plus "..." (e.g., "Hello worl..." for "Hello world")

The function works correctly for `max_length >= 3`, properly truncating to the specified length.

## Proposed Fix

```diff
def truncate_string(
    text: str,
    max_length: int = 100,
    normalize_whitespace: bool = False,
    keep_end: bool = False,
) -> str:
    if not text:
        return text

    if normalize_whitespace:
        text = re.sub(r"\s+", " ", text)

    if len(text) <= max_length:
        return text

    # Minimum sensible length for keep_end is 9 characters: "a... z"
    min_keep_end_length = 9

    if keep_end and max_length >= min_keep_end_length:
        # Calculate how much text to keep at each end
        # Subtract 5 for the "... " separator
        cutoff = (max_length - 5) // 2
        return text[:cutoff] + "... " + text[-cutoff:]
    else:
+       # For very small max_length, just truncate without ellipsis
+       if max_length < 3:
+           return text[:max_length]
        # Fall back to simple truncation for very small max_length
        return text[: max_length - 3] + "..."
```