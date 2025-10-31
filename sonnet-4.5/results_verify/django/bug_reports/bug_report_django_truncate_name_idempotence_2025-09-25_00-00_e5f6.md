# Bug Report: django.db.backends.base.schema.truncate_name Idempotence Violation

**Target**: `django.db.backends.base.schema.truncate_name`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `truncate_name` function is not idempotent: applying it twice with the same length can produce different results. A function that "shortens" identifiers should satisfy `truncate_name(truncate_name(x, L), L) == truncate_name(x, L)`, but this fails when `length < hash_len`.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from django.db.backends.base.schema import truncate_name

@given(st.text(min_size=1), st.integers(min_value=1, max_value=200))
def test_truncate_name_idempotence(identifier, length):
    truncated_once = truncate_name(identifier, length=length)
    truncated_twice = truncate_name(truncated_once, length=length)
    assert truncated_once == truncated_twice
```

**Failing input**: `identifier='00', length=1`

## Reproducing the Bug

```python
import django
from django.conf import settings

if not settings.configured:
    settings.configure(
        DATABASES={'default': {'ENGINE': 'django.db.backends.sqlite3', 'NAME': ':memory:'}},
        SECRET_KEY='test',
    )
    django.setup()

from django.db.backends.base.schema import truncate_name

once = truncate_name('00', length=1)
twice = truncate_name(once, length=1)

print(f"truncate_name('00', length=1) = '{once}'")
print(f"truncate_name('{once}', length=1) = '{twice}'")
print(f"Expected: {once} == {twice}")
print(f"Actual: {once} != {twice}")
```

## Why This Is A Bug

This bug is a consequence of the length invariant violation (see related bug report). Since `truncate_name` can produce results longer than `length`, applying it again produces a different (still-too-long) result.

Idempotence is a fundamental property of normalization functions. Users may call `truncate_name` multiple times in code paths, expecting consistent results. The violation indicates the function doesn't properly "settle" into a final form.

## Fix

This bug will be fixed by addressing the underlying length invariant bug:

```diff
--- a/django/db/backends/base/schema.py
+++ b/django/db/backends/base/schema.py
@@ -181,6 +181,9 @@ def truncate_name(identifier, length=None, hash_len=4):
     if length is None or len(name) <= length:
         return identifier

+    # Ensure we can fit at least the hash
+    hash_len = min(hash_len, length)
+
     digest = names_digest(name, length=hash_len)
     return "%s%s%s" % (
         '%s"."' % namespace if namespace else "",