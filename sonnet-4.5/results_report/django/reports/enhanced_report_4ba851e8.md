# Bug Report: django.db.models.sql.datastructures.Join Identity Excludes Critical Join Type

**Target**: `django.db.models.sql.datastructures.Join`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `Join.identity` property excludes critical fields (`join_type`, `table_alias`, and `nullable`) from the equality comparison, causing Join objects with different join types (INNER vs LEFT OUTER) to be considered equal and have identical hashes, violating the principle that equal objects should be functionally equivalent.

## Property-Based Test

```python
import django
from django.conf import settings

if not settings.configured:
    settings.configure(
        SECRET_KEY='test-key',
        INSTALLED_APPS=[],
        DATABASES={'default': {'ENGINE': 'django.db.backends.sqlite3', 'NAME': ':memory:'}}
    )

django.setup()

from hypothesis import given, strategies as st
from django.db.models.sql.datastructures import Join
from django.db.models.sql.constants import INNER, LOUTER


class MockField:
    def get_joining_columns(self):
        return [("id", "fk_id")]

    def get_extra_restriction(self, table_alias, parent_alias):
        return None


@given(st.text(min_size=1, max_size=20))
def test_join_type_affects_equality(table_name):
    field = MockField()

    join_inner = Join(
        table_name=table_name,
        parent_alias="t1",
        table_alias="t2",
        join_type=INNER,
        join_field=field,
        nullable=False
    )

    join_outer = Join(
        table_name=table_name,
        parent_alias="t1",
        table_alias="t2",
        join_type=LOUTER,
        join_field=field,
        nullable=False
    )

    assert join_inner.join_type != join_outer.join_type
    assert join_inner != join_outer, "INNER JOIN should not equal LEFT OUTER JOIN"
    if join_inner == join_outer:
        assert hash(join_inner) != hash(join_outer), "Different joins should have different hashes"


if __name__ == "__main__":
    test_join_type_affects_equality()
```

<details>

<summary>
**Failing input**: `table_name='0'` (or any other generated value)
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/7/hypo.py", line 55, in <module>
    test_join_type_affects_equality()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/7/hypo.py", line 27, in test_join_type_affects_equality
    def test_join_type_affects_equality(table_name):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/7/hypo.py", line 49, in test_join_type_affects_equality
    assert join_inner != join_outer, "INNER JOIN should not equal LEFT OUTER JOIN"
           ^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: INNER JOIN should not equal LEFT OUTER JOIN
Falsifying example: test_join_type_affects_equality(
    table_name='0',  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
import django
from django.conf import settings

if not settings.configured:
    settings.configure(
        SECRET_KEY='test-key',
        INSTALLED_APPS=[],
        DATABASES={'default': {'ENGINE': 'django.db.backends.sqlite3', 'NAME': ':memory:'}}
    )

django.setup()

from django.db.models.sql.datastructures import Join
from django.db.models.sql.constants import INNER, LOUTER


class MockField:
    def get_joining_columns(self):
        return [("id", "fk_id")]

    def get_extra_restriction(self, table_alias, parent_alias):
        return None


field = MockField()

join_inner = Join(
    table_name="users",
    parent_alias="t1",
    table_alias="t2",
    join_type=INNER,
    join_field=field,
    nullable=False
)

join_outer = Join(
    table_name="users",
    parent_alias="t1",
    table_alias="t2",
    join_type=LOUTER,
    join_field=field,
    nullable=False
)

print(f"join_inner.join_type = {join_inner.join_type}")
print(f"join_outer.join_type = {join_outer.join_type}")
print(f"join_inner == join_outer: {join_inner == join_outer}")
print(f"hash(join_inner) == hash(join_outer): {hash(join_inner) == hash(join_outer)}")

join_set = {join_inner, join_outer}
print(f"len({{join_inner, join_outer}}): {len(join_set)}")

# Additional debugging
print(f"\njoin_inner.identity = {join_inner.identity}")
print(f"join_outer.identity = {join_outer.identity}")
```

<details>

<summary>
Output demonstrating INNER JOIN and LEFT OUTER JOIN are incorrectly considered equal
</summary>
```
join_inner.join_type = INNER JOIN
join_outer.join_type = LEFT OUTER JOIN
join_inner == join_outer: True
hash(join_inner) == hash(join_outer): True
len({join_inner, join_outer}): 1

join_inner.identity = (<class 'django.db.models.sql.datastructures.Join'>, 'users', 't1', <__main__.MockField object at 0x76593074fb60>, None)
join_outer.identity = (<class 'django.db.models.sql.datastructures.Join'>, 'users', 't1', <__main__.MockField object at 0x76593074fb60>, None)
```
</details>

## Why This Is A Bug

The `identity` property in `/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/django/db/models/sql/datastructures.py:171-179` defines what makes two Join objects equal:

```python
@property
def identity(self):
    return (
        self.__class__,
        self.table_name,
        self.parent_alias,
        self.join_field,
        self.filtered_relation,
    )
```

This property is used by both `__eq__` (line 181-184) and `__hash__` (line 186-187) methods. The critical issue is that `join_type`, `table_alias`, and `nullable` are excluded from the identity tuple, meaning:

1. **INNER JOIN** and **LEFT OUTER JOIN** are considered equal despite producing fundamentally different SQL:
   - INNER JOIN: Returns only rows that have matching values in both tables
   - LEFT OUTER JOIN: Returns all rows from the left table plus matched rows from the right table, with NULLs for non-matching rows

2. When Join objects are stored in sets or used as dictionary keys (common in Django's query construction internals), only one join is retained when multiple "equal" joins with different types are added.

3. The `demote()` method (lines 189-192) changes `join_type` to INNER, and `promote()` method (lines 194-197) changes it to LOUTER, creating new objects that are considered equal to the original despite having different SQL semantics.

## Relevant Context

The Join class is critical to Django's ORM query construction, used extensively in `django.db.models.sql.Query` to build SQL JOIN clauses. The class documentation (lines 31-47) explicitly states that `join_type` is a required attribute for Join-compatible entries in the alias_map.

Django defines join types in `/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/django/db/models/sql/constants.py`:
- `INNER = "INNER JOIN"` (line 25)
- `LOUTER = "LEFT OUTER JOIN"` (line 26)

The `as_sql()` method (lines 88-152) uses `self.join_type` directly to generate SQL, confirming that different join types produce different SQL output.

## Proposed Fix

```diff
--- a/django/db/models/sql/datastructures.py
+++ b/django/db/models/sql/datastructures.py
@@ -171,10 +171,13 @@ class Join:
     @property
     def identity(self):
         return (
             self.__class__,
             self.table_name,
             self.parent_alias,
+            self.table_alias,
+            self.join_type,
             self.join_field,
+            self.nullable,
             self.filtered_relation,
         )
```