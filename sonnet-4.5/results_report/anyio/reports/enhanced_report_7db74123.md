# Bug Report: django.db.backends.base.operations Generates Invalid SQL with Negative LIMIT

**Target**: `django.db.backends.base.operations.BaseDatabaseOperations.limit_offset_sql`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `limit_offset_sql` method generates invalid SQL with negative LIMIT values when called with `high_mark < low_mark`, producing SQL like `LIMIT -5 OFFSET 10` which violates SQL standards.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

# Configure Django settings first to avoid circular imports
import django
from django.conf import settings

settings.configure(
    DEBUG=True,
    DATABASES={
        'default': {
            'ENGINE': 'django.db.backends.sqlite3',
            'NAME': ':memory:',
        }
    },
    INSTALLED_APPS=[],
    USE_TZ=False,
)
django.setup()

from unittest.mock import Mock
from hypothesis import given, strategies as st, settings, example
from django.db.backends.base.operations import BaseDatabaseOperations


@given(
    st.integers(min_value=0, max_value=10000),
    st.integers(min_value=0, max_value=10000)
)
@example(low_mark=10, high_mark=5)
@settings(max_examples=1000)
def test_limit_offset_sql_no_negative_limit(low_mark, high_mark):
    mock_conn = Mock()
    mock_conn.ops.no_limit_value.return_value = 2**63 - 1
    ops = BaseDatabaseOperations(connection=mock_conn)

    limit, offset = ops._get_limit_offset_params(low_mark, high_mark)

    assert limit is None or limit >= 0, \
        f"Negative limit: low_mark={low_mark}, high_mark={high_mark}, limit={limit}"


if __name__ == "__main__":
    test_limit_offset_sql_no_negative_limit()
```

<details>

<summary>
**Failing input**: `low_mark=10, high_mark=5`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/47/hypo.py", line 44, in <module>
    test_limit_offset_sql_no_negative_limit()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/47/hypo.py", line 27, in test_limit_offset_sql_no_negative_limit
    st.integers(min_value=0, max_value=10000),
               ^^^
  File "/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/hypothesis/core.py", line 2062, in wrapped_test
    _raise_to_user(errors, state.settings, [], " in explicit examples")
    ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/hypothesis/core.py", line 1613, in _raise_to_user
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/47/hypo.py", line 39, in test_limit_offset_sql_no_negative_limit
    assert limit is None or limit >= 0, \
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Negative limit: low_mark=10, high_mark=5, limit=-5
Falsifying explicit example: test_limit_offset_sql_no_negative_limit(
    low_mark=10,
    high_mark=5,
)
```
</details>

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

# Configure Django settings first to avoid circular imports
import django
from django.conf import settings

settings.configure(
    DEBUG=True,
    DATABASES={
        'default': {
            'ENGINE': 'django.db.backends.sqlite3',
            'NAME': ':memory:',
        }
    },
    INSTALLED_APPS=[],
    USE_TZ=False,
)
django.setup()

# Now we can import the operations module
from django.db.backends.base.operations import BaseDatabaseOperations
from unittest.mock import Mock

# Create a mock connection to avoid needing a real database
mock_conn = Mock()
mock_conn.ops.no_limit_value.return_value = 2**63 - 1
ops = BaseDatabaseOperations(connection=mock_conn)

# Call the public API with high_mark < low_mark
low_mark = 10
high_mark = 5

print(f"Input: low_mark={low_mark}, high_mark={high_mark}")
print()

# Generate SQL using the public limit_offset_sql method
sql = ops.limit_offset_sql(low_mark=low_mark, high_mark=high_mark)
print(f"Generated SQL clause: '{sql}'")
print()

# Also show the internal calculation
limit, offset = ops._get_limit_offset_params(low_mark=low_mark, high_mark=high_mark)
print(f"Internal calculation:")
print(f"  limit={limit} (computed as high_mark - offset = {high_mark} - {low_mark} = {limit})")
print(f"  offset={offset}")
print()

print(f"Issue: LIMIT {limit} is negative, which is invalid SQL.")
print("Most databases will reject this query with a syntax error.")
```

<details>

<summary>
Generated invalid SQL with negative LIMIT
</summary>
```
Input: low_mark=10, high_mark=5

Generated SQL clause: 'LIMIT -5 OFFSET 10'

Internal calculation:
  limit=-5 (computed as high_mark - offset = 5 - 10 = -5)
  offset=10

Issue: LIMIT -5 is negative, which is invalid SQL.
Most databases will reject this query with a syntax error.
```
</details>

## Why This Is A Bug

This violates expected behavior because `limit_offset_sql` is a public API method (no leading underscore) that generates syntactically invalid SQL when `high_mark < low_mark`. The SQL standard requires LIMIT to be a non-negative integer. The generated SQL `LIMIT -5 OFFSET 10` is invalid and will cause syntax errors in most databases.

The bug occurs in the `_get_limit_offset_params` method at line 262 of `/django/db/backends/base/operations.py`:
```python
return (high_mark - offset), offset
```
When `high_mark=5` and `offset=10`, this calculates `(5 - 10) = -5`, resulting in a negative LIMIT value.

While Django's internal Query class prevents this scenario through protective logic in `set_limits()` (lines 2168-2169 of query.py), the method itself is a public API that could be called by Django extensions, custom database backends, or third-party code. Public APIs should follow defensive programming principles and either validate inputs or handle edge cases gracefully.

## Relevant Context

- The method `limit_offset_sql` is defined at line 267-277 in `/django/db/backends/base/operations.py`
- The internal calculation happens in `_get_limit_offset_params` at lines 259-265
- Django's Query class has protective logic that calls `set_empty()` when `low_mark == high_mark`, but doesn't prevent all cases where `high_mark < low_mark`
- Different databases handle negative LIMIT values inconsistently - SQLite treats it as unlimited, PostgreSQL rejects it with a syntax error
- The method's docstring (line 268) simply states "Return LIMIT/OFFSET SQL clause" without documenting expected behavior for edge cases
- Source code location: [django/db/backends/base/operations.py:259-277](https://github.com/django/django/blob/main/django/db/backends/base/operations.py#L259-L277)

## Proposed Fix

Add validation to ensure non-negative LIMIT values by clamping the result:

```diff
--- a/django/db/backends/base/operations.py
+++ b/django/db/backends/base/operations.py
@@ -259,7 +259,7 @@ class BaseDatabaseOperations:
     def _get_limit_offset_params(self, low_mark, high_mark):
         offset = low_mark or 0
         if high_mark is not None:
-            return (high_mark - offset), offset
+            return (max(0, high_mark - offset)), offset
         elif offset:
             return self.connection.ops.no_limit_value(), offset
         return None, offset
```