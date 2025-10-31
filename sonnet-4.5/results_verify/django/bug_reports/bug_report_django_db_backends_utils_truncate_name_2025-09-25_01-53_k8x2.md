# Bug Report: django.db.backends.utils.truncate_name Length Violation

**Target**: `django.db.backends.utils.truncate_name`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `truncate_name` function violates its documented contract when `length < hash_len` (default 4). It returns identifiers longer than the requested `length`, which can cause issues when database systems enforce strict identifier length limits.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from django.db.backends.utils import truncate_name, split_identifier

@given(st.text(min_size=1, max_size=200), st.integers(min_value=1, max_value=50))
def test_truncate_name_respects_length(identifier, length):
    result = truncate_name(identifier, length=length)
    namespace, name = split_identifier(result)
    actual_name_length = len(name)
    assert actual_name_length <= length
```

**Failing input**: `identifier='00', length=1`

## Reproducing the Bug

```python
from django.db.backends.utils import truncate_name, split_identifier

identifier = '00'
length = 1

result = truncate_name(identifier, length=length)
_, name = split_identifier(result)

print(f"truncate_name('{identifier}', length={length})")
print(f"Expected: name with length <= {length}")
print(f"Actual: '{result}' (length = {len(name)})")
```

Output:
```
truncate_name('00', length=1)
Expected: name with length <= 1
Actual: 'd3d9' (length = 4)
Bug: The function returns a name of length 4, which exceeds the requested length of 1
```

## Why This Is A Bug

The function's docstring states it should "Shorten an SQL identifier to a repeatable mangled version with the given length." However, when `length < hash_len` (default 4), the function returns an identifier that exceeds the requested length.

The problem is on line 299:
```python
name[: length - hash_len]
```

When `length=1` and `hash_len=4`, this becomes `name[:-3]`, which for short names results in an empty string or very short prefix. The digest is then appended, making the final result at least `hash_len` characters long, violating the length constraint.

## Fix

```diff
--- a/django/db/backends/utils.py
+++ b/django/db/backends/utils.py
@@ -290,6 +290,10 @@ def truncate_name(identifier, length=None, hash_len=4):
     """
     namespace, name = split_identifier(identifier)

+    if length is not None and length < hash_len:
+        # When requested length is less than hash_len, use the full length for hashing
+        hash_len = length
+
     if length is None or len(name) <= length:
         return identifier

```

This ensures that when `length < hash_len`, we adjust `hash_len` to match `length`, so the final truncated name respects the length constraint.