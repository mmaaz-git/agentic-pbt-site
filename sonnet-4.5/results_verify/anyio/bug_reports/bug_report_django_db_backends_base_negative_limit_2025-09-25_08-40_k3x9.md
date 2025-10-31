# Bug Report: django.db.backends.base.operations negative LIMIT generation

**Target**: `django.db.backends.base.operations.BaseDatabaseOperations.limit_offset_sql`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `limit_offset_sql` method can generate invalid SQL with negative LIMIT values when called with `high_mark < low_mark`, producing SQL like `LIMIT -5 OFFSET 10` which is invalid in SQL.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

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
```

**Failing input**: `low_mark=10, high_mark=5`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from unittest.mock import Mock
from django.db.backends.base.operations import BaseDatabaseOperations

mock_conn = Mock()
mock_conn.ops.no_limit_value.return_value = 2**63 - 1
ops = BaseDatabaseOperations(connection=mock_conn)

sql = ops.limit_offset_sql(low_mark=10, high_mark=5)
print(f"Generated SQL: {sql}")

limit, offset = ops._get_limit_offset_params(low_mark=10, high_mark=5)
print(f"limit={limit}, offset={offset}")
```

Output:
```
Generated SQL: 'LIMIT -5 OFFSET 10'
limit=-5, offset=10
```

## Why This Is A Bug

The `limit_offset_sql` method is a public API (no leading underscore) that can be called by subclasses or potentially external code. When called with `high_mark < low_mark`, it generates SQL with a negative LIMIT value (e.g., `LIMIT -5 OFFSET 10`), which is invalid SQL.

While the Django ORM's Query class prevents this through its `set_limits` method (lines 2168-2169 in query.py check for `low_mark == high_mark` and call `set_empty()`), the function itself doesn't validate its inputs. This violates defensive programming principles for a public API.

The calculation in `_get_limit_offset_params` at line 262 is:
```python
return (high_mark - offset), offset
```

When `high_mark=5` and `offset=10`, this returns `(-5, 10)`, which is then used to generate invalid SQL.

## Fix

Add validation in `_get_limit_offset_params` to ensure high_mark >= low_mark when both are provided:

```diff
--- a/django/db/backends/base/operations.py
+++ b/django/db/backends/base/operations.py
@@ -259,6 +259,8 @@ class BaseDatabaseOperations:
     def _get_limit_offset_params(self, low_mark, high_mark):
         offset = low_mark or 0
         if high_mark is not None:
+            if high_mark < offset:
+                raise ValueError(f"high_mark ({high_mark}) must be >= low_mark ({low_mark})")
             return (high_mark - offset), offset
         elif offset:
             return self.connection.ops.no_limit_value(), offset
```

Alternatively, clamp the limit to a minimum of 0:

```diff
--- a/django/db/backends/base/operations.py
+++ b/django/db/backends/base/operations.py
@@ -259,7 +259,7 @@ class BaseDatabaseOperations:
     def _get_limit_offset_params(self, low_mark, high_mark):
         offset = low_mark or 0
         if high_mark is not None:
-            return (high_mark - offset), offset
+            return (max(0, high_mark - offset), offset)
         elif offset:
             return self.connection.ops.no_limit_value(), offset
         return None, offset
```