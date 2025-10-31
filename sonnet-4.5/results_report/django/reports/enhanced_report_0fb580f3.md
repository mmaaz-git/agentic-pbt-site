# Bug Report: django.db.backends.utils.truncate_name Length Contract Violation

**Target**: `django.db.backends.utils.truncate_name`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `truncate_name` function violates its documented contract by producing output longer than the specified `length` parameter when `hash_len >= length`, due to incorrect handling of the truncation logic.

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

if __name__ == "__main__":
    test_truncate_name_length_invariant()
```

<details>

<summary>
**Failing input**: `identifier='00', length=1, hash_len=2`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/34/hypo.py", line 15, in <module>
    test_truncate_name_length_invariant()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/34/hypo.py", line 5, in test_truncate_name_length_invariant
    @settings(max_examples=1000)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/34/hypo.py", line 12, in test_truncate_name_length_invariant
    assert name_length <= length, f"Truncated name '{name}' has length {name_length} > {length}"
           ^^^^^^^^^^^^^^^^^^^^^
AssertionError: Truncated name '0b4' has length 3 > 1
Falsifying example: test_truncate_name_length_invariant(
    identifier='00',
    length=1,
    hash_len=2,
)
```
</details>

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
print(f"Bug: Output length ({len(result)}) exceeds requested length ({length})")
```

<details>

<summary>
Output length exceeds requested length
</summary>
```
Input: identifier='00', length=1, hash_len=2
Output: '0b4'
Output length: 3
Expected: output length <= 1
Actual: output length = 3
Bug: Output length (3) exceeds requested length (1)
```
</details>

## Why This Is A Bug

The function's docstring explicitly states it will "Shorten an SQL identifier to a repeatable mangled version with the given length." However, when `hash_len >= length`, the function produces output that exceeds the requested `length`.

The bug occurs in the truncation logic at line 299: `name[: length - hash_len]`. When `hash_len >= length`, this slice becomes `name[: negative_number]` which returns an empty string in Python. The function then concatenates this empty string with the digest (which has length `hash_len`), resulting in output of length `hash_len` instead of the requested `length`.

This violates the principle of least surprise - callers reasonably expect that `truncate_name(x, length=n)` will never produce output longer than `n` characters. The function provides no validation, documentation of preconditions, or error handling for this case.

## Relevant Context

The `truncate_name` function is critical for Django's database operations, particularly with Oracle databases which have strict 30-character identifier limits. It's used in:

- Model table name generation (`django/db/models/options.py:209`)
- Many-to-many table names (`django/db/models/fields/related.py`)
- Oracle sequence names and identifiers (`django/db/backends/oracle/operations.py`)
- Database schema migration operations (`django/db/backends/base/schema.py`)

While the default usage patterns (e.g., Oracle with `length=30, hash_len=4`) avoid this bug, any code that dynamically sets these parameters or uses smaller length values could encounter incorrect behavior leading to database errors.

Django's source code: https://github.com/django/django/blob/main/django/db/backends/utils.py#L283

## Proposed Fix

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

+   # When hash_len >= length, we can't include any of the original name
+   # Just use the digest truncated to the requested length
+   if hash_len >= length:
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