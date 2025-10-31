# Bug Report: django.db.backends.utils.truncate_name Length Contract Violation

**Target**: `django.db.backends.utils.truncate_name`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `truncate_name` function violates its documented contract by returning identifiers longer than the requested `length` parameter when `length < hash_len` (default 4), potentially causing database errors when strict identifier length limits are enforced.

## Property-Based Test

```python
#!/usr/bin/env python3
"""
Property-based test for django.db.backends.utils.truncate_name
This test verifies that the function respects its length parameter constraint.
"""

import sys
import os

# Add Django to path
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings
from django.db.backends.utils import truncate_name, split_identifier

@given(st.text(min_size=1, max_size=200), st.integers(min_value=1, max_value=50))
@settings(max_examples=100)
def test_truncate_name_respects_length(identifier, length):
    """
    Test that truncate_name always returns an identifier with the name portion
    having length <= the requested length parameter.
    """
    result = truncate_name(identifier, length=length)
    namespace, name = split_identifier(result)
    actual_name_length = len(name)

    # The property that should hold: name length should not exceed requested length
    assert actual_name_length <= length, (
        f"truncate_name('{identifier}', length={length}) returned '{result}' "
        f"with name length {actual_name_length}, exceeding requested length {length}"
    )

if __name__ == "__main__":
    # Run the property-based test
    print("Running property-based test for truncate_name...")
    print("This test verifies that truncate_name respects its length parameter.")
    print("="*60)

    try:
        test_truncate_name_respects_length()
        print("✓ All tests passed!")
    except AssertionError as e:
        print(f"❌ Test failed!")
        print(f"Assertion error: {e}")
    except Exception as e:
        print(f"❌ Test encountered an error!")
        print(f"Error details: {e}")
```

<details>

<summary>
**Failing input**: `identifier='00', length=1`
</summary>
```
Running property-based test for truncate_name...
This test verifies that truncate_name respects its length parameter.
============================================================
❌ Test failed!
Assertion error: truncate_name('00', length=1) returned 'b4b1' with name length 4, exceeding requested length 1
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""
Minimal reproduction of the truncate_name bug in Django.
This demonstrates that the function violates its contract when length < hash_len.
"""

import sys
import os

# Add Django to path
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from django.db.backends.utils import truncate_name, split_identifier

# Test case from the bug report
identifier = '00'
length = 1

print(f"Testing truncate_name('{identifier}', length={length})")
print(f"Expected: Result with length <= {length}")

result = truncate_name(identifier, length=length)
namespace, name = split_identifier(result)

print(f"Actual result: '{result}'")
print(f"Name part: '{name}' (length = {len(name)})")

if len(name) > length:
    print(f"\n❌ BUG CONFIRMED: The function returns a name of length {len(name)}, which exceeds the requested length of {length}")
else:
    print(f"\n✓ OK: Result respects the length constraint")

# Additional test cases to demonstrate the bug pattern
print("\n" + "="*60)
print("Additional test cases:")
print("="*60)

test_cases = [
    ('abc', 1),
    ('test', 2),
    ('database_name', 3),
    ('very_long_identifier_name', 4),
    ('short', 5),
]

for test_id, test_length in test_cases:
    result = truncate_name(test_id, length=test_length)
    _, name_part = split_identifier(result)
    status = "❌ BUG" if len(name_part) > test_length else "✓ OK"
    print(f"truncate_name('{test_id}', {test_length}) -> '{result}' (name length={len(name_part)}) {status}")
```

<details>

<summary>
truncate_name violates length constraint, returning 4-character result when 1 character requested
</summary>
```
Testing truncate_name('00', length=1)
Expected: Result with length <= 1
Actual result: 'b4b1'
Name part: 'b4b1' (length = 4)

❌ BUG CONFIRMED: The function returns a name of length 4, which exceeds the requested length of 1

============================================================
Additional test cases:
============================================================
truncate_name('abc', 1) -> '9001' (name length=4) ❌ BUG
truncate_name('test', 2) -> 'te098f' (name length=6) ❌ BUG
truncate_name('database_name', 3) -> 'database_namded7' (name length=16) ❌ BUG
truncate_name('very_long_identifier_name', 4) -> 'cb2a' (name length=4) ✓ OK
truncate_name('short', 5) -> 'short' (name length=5) ✓ OK
```
</details>

## Why This Is A Bug

The function's docstring explicitly states it should "Shorten an SQL identifier to a repeatable mangled version with the given length." This creates a clear contract that the returned identifier should not exceed the specified `length` parameter.

The bug occurs in line 299 of `/django/db/backends/utils.py`:
```python
name[: length - hash_len]
```

When `length < hash_len` (default 4), the slice operation produces unexpected results:
- For `length=1, hash_len=4`: `name[:1-4]` = `name[:-3]`
- For a short identifier like '00', this results in an empty string
- The function then concatenates: empty string + 4-character hash = 4 characters total
- This violates the contract since 4 > 1 (requested length)

The bug manifests whenever:
1. The identifier needs truncation (`len(name) > length`)
2. The requested `length < hash_len` (typically when length < 4)

This is particularly problematic because:
- Django uses `truncate_name` throughout its database backends to enforce database-specific identifier length limits
- Database systems strictly enforce these limits and will reject identifiers that exceed them
- If a database requires identifiers ≤ N characters and `truncate_name(identifier, N)` returns > N characters, it will cause database errors

## Relevant Context

The `truncate_name` function is critical infrastructure used across Django's database layer:

1. **Oracle Backend** (`django/db/backends/oracle/`): Uses `truncate_name` with `max_name_length()=30` to comply with Oracle's strict 30-character identifier limit
2. **PostgreSQL Backend**: Uses it for constraint and index names (63-character limit)
3. **Schema Editor**: Relies on it to generate valid index, constraint, and foreign key names

The function uses MD5 hashing to create reproducible truncated names, which is important for database migrations and schema consistency. The hash digest ensures that the same long identifier always gets truncated to the same short identifier across different runs.

While `length < 4` is rare in practice (most database limits are 30+ characters), the function should still honor its documented contract for correctness and reliability.

## Proposed Fix

```diff
--- a/django/db/backends/utils.py
+++ b/django/db/backends/utils.py
@@ -290,6 +290,10 @@ def truncate_name(identifier, length=None, hash_len=4):
     """
     namespace, name = split_identifier(identifier)

+    # Ensure hash_len doesn't exceed the requested length
+    if length is not None and hash_len > length:
+        hash_len = length
+
     if length is None or len(name) <= length:
         return identifier

```