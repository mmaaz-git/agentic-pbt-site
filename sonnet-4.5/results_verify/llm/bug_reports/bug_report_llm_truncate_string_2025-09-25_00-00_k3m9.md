# Bug Report: llm.utils.truncate_string Length Constraint Violation

**Target**: `llm.utils.truncate_string`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `truncate_string` function violates its core contract by returning strings longer than `max_length` when `max_length < 3`.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from llm.utils import truncate_string

@given(
    st.text(min_size=1, max_size=1000),
    st.integers(min_value=1, max_value=500)
)
def test_truncate_string_length_constraint(text, max_length):
    result = truncate_string(text, max_length)
    assert len(result) <= max_length, f"Result length {len(result)} > max_length {max_length}"
```

**Failing input**: `text = "hello", max_length = 1`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

from llm.utils import truncate_string

text = "hello"
max_length = 1
result = truncate_string(text, max_length)

print(f"Input: '{text}' (length {len(text)})")
print(f"Max length: {max_length}")
print(f"Result: '{result}' (length {len(result)})")
print(f"Constraint violated: {len(result)} > {max_length}")
```

Expected output:
```
Input: 'hello' (length 5)
Max length: 1
Result: 'h' (length 1)
Constraint violated: False
```

Actual output (based on code analysis):
```
Input: 'hello' (length 5)
Max length: 1
Result: '...' (length 3)
Constraint violated: True
```

## Why This Is A Bug

Looking at the code (lines 439-476 in utils.py):

```python
if keep_end and max_length >= min_keep_end_length:
    cutoff = (max_length - 5) // 2
    return text[:cutoff] + "... " + text[-cutoff:]
else:
    return text[: max_length - 3] + "..."  # BUG: When max_length < 3, this fails
```

When `max_length = 1`:
- Returns `text[:1-3] + "..."` = `text[:-2] + "..."`
- This appends "..." (3 chars) regardless of the negative slice
- Result is always at least 3 characters, violating the constraint

When `max_length = 2`:
- Returns `text[:2-3] + "..."` = `text[:-1] + "..."`
- Still results in more than 2 characters

The function's docstring states: "Truncate a string to a maximum length" but fails to enforce this contract for `max_length < 3`.

## Fix

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

+   # Handle edge case where max_length is too small for ellipsis
+   if max_length < 3:
+       return text[:max_length]

    min_keep_end_length = 9

    if keep_end and max_length >= min_keep_end_length:
        cutoff = (max_length - 5) // 2
        return text[:cutoff] + "... " + text[-cutoff:]
    else:
        return text[: max_length - 3] + "..."
```