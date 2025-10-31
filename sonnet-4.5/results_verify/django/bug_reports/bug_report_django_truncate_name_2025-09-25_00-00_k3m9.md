# Bug Report: django.db.backends.utils.truncate_name Length Limit Violation

**Target**: `django.db.backends.utils.truncate_name`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `truncate_name` function violates its contract by returning identifiers longer than the specified `length` parameter when `length < hash_len` (default hash_len=4). This breaks the fundamental property that the truncated result should never exceed the requested length.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from django.db.backends.utils import truncate_name

@given(st.text(), st.integers(min_value=1, max_value=200))
def test_truncate_name_length_invariant(identifier, length):
    result = truncate_name(identifier, length)
    from django.db.backends.utils import split_identifier
    namespace, name = split_identifier(result)
    if namespace:
        assert len(name) <= length
    else:
        assert len(result) <= length
```

**Failing input**: `identifier='00', length=1`

## Reproducing the Bug

```python
from django.db.backends.utils import truncate_name

identifier = '00'
length = 1

result = truncate_name(identifier, length)

print(f"truncate_name({identifier!r}, {length}) = {result!r}")
print(f"Result length: {len(result)}")
print(f"Expected max length: {length}")
assert len(result) <= length
```

Output:
```
truncate_name('00', 1) = 'b4b1'
Result length: 4
Expected max length: 1
AssertionError
```

## Why This Is A Bug

The function's docstring states it will "Shorten an SQL identifier to a repeatable mangled version with the given length." This implies the result should never exceed `length` characters. However, when `length < hash_len`, the function returns a string that is `hash_len` characters long, violating this contract.

The bug occurs because the code uses `name[: length - hash_len]` which becomes a negative index when `length < hash_len`. This produces a truncated name portion, but the final result still includes the full `hash_len`-character digest, making the total length equal to approximately `hash_len` characters rather than `length` characters.

This also breaks idempotence: calling `truncate_name(truncate_name('00', 1), 1)` produces a different result than the first call.

## Fix

```diff
--- a/django/db/backends/utils.py
+++ b/django/db/backends/utils.py
@@ -290,6 +290,10 @@ def truncate_name(identifier, length=None, hash_len=4):
     """
     namespace, name = split_identifier(identifier)

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