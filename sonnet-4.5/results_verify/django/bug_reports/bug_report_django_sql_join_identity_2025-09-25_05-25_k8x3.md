# Bug Report: django.db.models.sql Join Identity Incomplete

**Target**: `django.db.models.sql.datastructures.Join`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `Join.identity` property excludes `join_type`, `table_alias`, and `nullable` fields, causing Join objects with different join types (INNER vs LEFT OUTER) to be considered equal and have identical hashes. This violates the principle that equal objects should be functionally equivalent, as INNER JOIN and LEFT OUTER JOIN produce different SQL results.

## Property-Based Test

```python
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
```

**Failing input**: Any table name (e.g., `"users"`)

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
```

**Expected output**:
```
join_inner.join_type = INNER JOIN
join_outer.join_type = LEFT OUTER JOIN
join_inner == join_outer: False
hash(join_inner) == hash(join_outer): False
len({join_inner, join_outer}): 2
```

**Actual output**:
```
join_inner.join_type = INNER JOIN
join_outer.join_type = LEFT OUTER JOIN
join_inner == join_outer: True
hash(join_inner) == hash(join_outer): True
len({join_inner, join_outer}): 1
```

## Why This Is A Bug

The `identity` property (lines 171-179 in `django/db/models/sql/datastructures.py`) defines what makes two Join objects equal:

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

This excludes `join_type`, meaning:
1. **INNER JOIN** and **LEFT OUTER JOIN** are considered equal
2. These produce fundamentally different SQL and results:
   - INNER JOIN: Only matching rows from both tables
   - LEFT OUTER JOIN: All rows from left table, NULLs for non-matches
3. When added to a set/dict, only one join is retained (the first one)
4. The `demote()` and `promote()` methods create "equal" objects with different semantics

This also affects `table_alias` and `nullable`, which are similarly excluded from identity.

## Impact

- **Query Construction**: If joins are stored in sets or used as dict keys, duplicates with different join types will be incorrectly deduplicated
- **Caching**: Cache keys based on join equality may collide inappropriately
- **demote()/promote()**: These methods create objects that are equal to the original despite having different SQL behavior

## Fix

```diff
--- a/django/db/models/sql/datastructures.py
+++ b/django/db/models/sql/datastructures.py
@@ -170,10 +170,12 @@ class Join:

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

This ensures that Join objects are only considered equal when they are truly functionally equivalent.