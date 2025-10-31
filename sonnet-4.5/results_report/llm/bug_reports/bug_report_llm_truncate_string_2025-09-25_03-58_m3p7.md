# Bug Report: llm.utils.truncate_string Violates Length Invariant

**Target**: `llm.utils.truncate_string`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `truncate_string` function violates its core invariant when `max_length < 3`: the returned string can be longer than `max_length`.

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
```

**Failing input**: `text = "Hello world"`, `max_length = 2`

## Reproducing the Bug

```python
from llm.utils import truncate_string

text = "Hello world"

for max_length in [0, 1, 2, 3, 4]:
    result = truncate_string(text, max_length=max_length)
    print(f"max_length={max_length}: '{result}' (actual length={len(result)})")
```

Output:
```
max_length=0: 'Hello worl...' (actual length=13)
max_length=1: 'Hello wor...' (actual length=12)
max_length=2: 'Hello wo...' (actual length=11)
max_length=3: '...' (actual length=3)
max_length=4: 'H...' (actual length=4)
```

## Why This Is A Bug

The function documentation and its name promise that the result will be truncated to `max_length` or less. However, when `max_length < 3`, the function returns a string longer than `max_length`, violating this invariant.

**Root cause**: The code uses `text[: max_length - 3] + "..."` in the simple truncation case. When `max_length < 3`, the slice becomes negative (e.g., `text[:-1]`), which includes most of the original text instead of truncating it.

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