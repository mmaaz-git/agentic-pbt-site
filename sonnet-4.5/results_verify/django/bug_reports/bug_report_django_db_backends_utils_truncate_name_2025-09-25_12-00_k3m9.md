# Bug Report: django.db.backends.utils.truncate_name Length Invariant Violation

**Target**: `django.db.backends.utils.truncate_name`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `truncate_name` function violates its documented contract of shortening an identifier to "the given length" when `hash_len >= length`. The function produces identifiers longer than the specified `length` parameter.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from django.db.backends.utils import truncate_name, split_identifier

@given(st.text(min_size=1), st.integers(min_value=1, max_value=1000), st.integers(min_value=1, max_value=10))
@settings(max_examples=1000)
def test_truncate_name_length_invariant(identifier, length, hash_len):
    result = truncate_name(identifier, length=length, hash_len=hash_len)

    namespace, name = split_identifier(result)
    name_length = len(name)

    assert name_length <= length, f"Truncated name '{name}' has length {name_length} > {length}"
```

**Failing input**: `identifier='00', length=1, hash_len=2`

## Reproducing the Bug

```python
from django.db.backends.utils import truncate_name

identifier = '00'
length = 1
hash_len = 2

result = truncate_name(identifier, length=length, hash_len=hash_len)

print(f"Input: identifier={identifier!r}, length={length}, hash_len={hash_len}")
print(f"Output: {result!r}")
print(f"Output length: {len(result)}")
print(f"Expected: output length <= {length}")
print(f"Actual: output length = {len(result)}")
```

**Output:**
```
Input: identifier='00', length=1, hash_len=2
Output: '0b4'
Output length: 3
Expected: output length <= 1
Actual: output length = 3
```

## Why This Is A Bug

1. **Contract Violation**: The docstring states the function will "Shorten an SQL identifier to a repeatable mangled version with the given length", but when `hash_len >= length`, the output exceeds the specified `length`.

2. **No Validation**: The function doesn't validate that `hash_len < length` and doesn't raise an error when this precondition is violated.

3. **Silent Failure**: The function silently produces incorrect output rather than failing fast or handling the edge case gracefully.

4. **Violates User Expectations**: Callers have a reasonable expectation that `len(truncate_name(x, length=n)) <= n` for all valid inputs.

## Fix

```diff
def truncate_name(identifier, length=None, hash_len=4):
    """
    Shorten an SQL identifier to a repeatable mangled version with the given
    length.

    If a quote stripped name contains a namespace, e.g. USERNAME"."TABLE,
    truncate the table portion only.
    """
    namespace, name = split_identifier(identifier)

    if length is None or len(name) <= length:
        return identifier

+   if hash_len >= length:
+       # Can't meaningfully truncate with hash_len >= length, use only digest
+       digest = names_digest(name, length=length)
+       return "%s%s" % (
+           '%s"."' % namespace if namespace else "",
+           digest,
+       )

    digest = names_digest(name, length=hash_len)
    return "%s%s%s" % (
        '%s"."' % namespace if namespace else "",
        name[: length - hash_len],
        digest,
    )
```