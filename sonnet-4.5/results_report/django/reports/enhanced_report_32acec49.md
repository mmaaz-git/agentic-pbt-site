# Bug Report: django.db.backends.utils.truncate_name Violates Length Contract

**Target**: `django.db.backends.utils.truncate_name`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `truncate_name` function returns identifiers longer than the specified `length` parameter when `length < hash_len` (default hash_len=4), violating its documented contract to produce output "with the given length".

## Property-Based Test

```python
#!/usr/bin/env python3
"""Property-based test for django.db.backends.utils.truncate_name bug"""

import sys
import os
# Add Django to path
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings
from django.db.backends.utils import truncate_name, split_identifier

@given(st.text(), st.integers(min_value=1, max_value=200))
def test_truncate_name_length_invariant(identifier, length):
    result = truncate_name(identifier, length)
    namespace, name = split_identifier(result)
    if namespace:
        assert len(name) <= length, f"Name portion {name!r} has length {len(name)} > {length}"
    else:
        assert len(result) <= length, f"Result {result!r} has length {len(result)} > {length}"

# Run the test
if __name__ == "__main__":
    # Use explicit settings for reproducibility
    test_truncate_name_length_invariant()
```

<details>

<summary>
**Failing input**: `identifier='00', length=1`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/12/hypo.py", line 24, in <module>
    test_truncate_name_length_invariant()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/12/hypo.py", line 13, in test_truncate_name_length_invariant
    def test_truncate_name_length_invariant(identifier, length):
                   ^^^
  File "/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/12/hypo.py", line 19, in test_truncate_name_length_invariant
    assert len(result) <= length, f"Result {result!r} has length {len(result)} > {length}"
           ^^^^^^^^^^^^^^^^^^^^^
AssertionError: Result 'b4b1' has length 4 > 1
Falsifying example: test_truncate_name_length_invariant(
    identifier='00',
    length=1,
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/django/db/backends/utils.py:296
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""Minimal reproduction of django.db.backends.utils.truncate_name bug"""

import sys
import os
# Add Django to path
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from django.db.backends.utils import truncate_name

# Test case that should fail
identifier = '00'
length = 1

result = truncate_name(identifier, length)

print(f"truncate_name({identifier!r}, {length}) = {result!r}")
print(f"Result length: {len(result)}")
print(f"Expected max length: {length}")
print(f"Violates contract: {len(result) > length}")

# Test idempotence issue
result2 = truncate_name(result, length)
print(f"\nIdempotence test:")
print(f"First call:  truncate_name({identifier!r}, {length}) = {result!r}")
print(f"Second call: truncate_name({result!r}, {length}) = {result2!r}")
print(f"Is idempotent: {result == result2}")

# Show the assertion failure
try:
    assert len(result) <= length, f"Result length {len(result)} exceeds requested length {length}"
except AssertionError as e:
    print(f"\nAssertion Error: {e}")
```

<details>

<summary>
AssertionError: Result length 4 exceeds requested length 1
</summary>
```
truncate_name('00', 1) = 'b4b1'
Result length: 4
Expected max length: 1
Violates contract: True

Idempotence test:
First call:  truncate_name('00', 1) = 'b4b1'
Second call: truncate_name('b4b1', 1) = 'bfde7'
Is idempotent: False

Assertion Error: Result length 4 exceeds requested length 1
```
</details>

## Why This Is A Bug

The function's docstring explicitly states it will "Shorten an SQL identifier to a repeatable mangled version with the given length." This creates a clear contract that the output length should not exceed the `length` parameter. However, when `length < hash_len` (default 4), the function violates this contract by returning a result that is `hash_len` characters long instead of `length` characters.

The bug occurs in line 299 of `/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/django/db/backends/utils.py`:
```python
name[: length - hash_len]
```

When `length=1` and `hash_len=4`, this becomes `name[:-3]`. For `name='00'`, this produces an empty string. The function then concatenates an empty namespace prefix, empty truncated name, and the full 4-character hash digest, resulting in a 4-character output that violates the requested 1-character limit.

This also breaks idempotence: calling `truncate_name(truncate_name('00', 1), 1)` produces 'bfde7' instead of 'b4b1', making the function non-deterministic for repeated truncations.

## Relevant Context

Django's `truncate_name` function is used internally by the database backend layer to ensure SQL identifiers comply with database-specific length limits:
- PostgreSQL: 63 characters
- Oracle: 30 characters
- MySQL: 64 characters

While these real-world limits are much larger than where the bug manifests (length < 4), the function should still honor its documented contract for all valid inputs. The function accepts any positive length value without raising an error, implying it should handle edge cases correctly.

The function uses MD5 hashing (via `names_digest`) to create a reproducible short suffix that helps avoid naming collisions when truncating identifiers. This is important for database migrations and schema management.

Source code location: [django/db/backends/utils.py:283-301](https://github.com/django/django/blob/main/django/db/backends/utils.py#L283-L301)

## Proposed Fix

```diff
--- a/django/db/backends/utils.py
+++ b/django/db/backends/utils.py
@@ -293,6 +293,11 @@ def truncate_name(identifier, length=None, hash_len=4):
     if length is None or len(name) <= length:
         return identifier

+    # If length is too small to accommodate the hash, just truncate without hash
+    if length <= hash_len:
+        return "%s%s" % (
+            '%s"."' % namespace if namespace else "",
+            name[:length],
+        )
+
     digest = names_digest(name, length=hash_len)
     return "%s%s%s" % (
         '%s"."' % namespace if namespace else "",
         name[: length - hash_len],
         digest,
     )
```