# Bug Report: django.db.backends.postgresql date_extract_sql accepts invalid PostgreSQL field names

**Target**: `django.db.backends.postgresql.operations.DatabaseOperations.date_extract_sql`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `date_extract_sql` method accepts invalid PostgreSQL EXTRACT field names (like "WEEK_DAY", "ISO_WEEK_DAY", "ISO_YEAR") that will cause runtime failures when executed against PostgreSQL. The regex validation `[A-Z_]+` is too permissive and doesn't ensure the field name is valid in PostgreSQL.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from django.db.backends.postgresql.operations import DatabaseOperations

class MockConnection:
    pass

ops = DatabaseOperations(MockConnection())

VALID_PG_EXTRACT_FIELDS = {
    'CENTURY', 'DAY', 'DECADE', 'DOW', 'DOY', 'EPOCH', 'HOUR',
    'ISODOW', 'ISOYEAR', 'JULIAN', 'MICROSECONDS', 'MILLENNIUM',
    'MILLISECONDS', 'MINUTE', 'MONTH', 'QUARTER', 'SECOND',
    'TIMEZONE', 'TIMEZONE_HOUR', 'TIMEZONE_MINUTE', 'WEEK', 'YEAR'
}

@given(st.text(min_size=1).filter(lambda s: s.isupper() and '_' in s))
def test_date_extract_rejects_invalid_uppercase_fields(lookup_type):
    if lookup_type not in VALID_PG_EXTRACT_FIELDS:
        try:
            sql, params = ops.date_extract_sql(lookup_type, 'test_column', ())
            assert False, f"Should reject invalid field: {lookup_type}"
        except ValueError:
            pass
```

**Failing input**: `"WEEK_DAY"`, `"ISO_WEEK_DAY"`, `"ISO_YEAR"`, `"INVALID_FIELD"`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django.conf.global_settings')

import django
from django.conf import settings
if not settings.configured:
    settings.configure(
        DATABASES={'default': {'ENGINE': 'django.db.backends.postgresql', 'NAME': 'test'}},
        USE_TZ=True,
    )
    django.setup()

from django.db.backends.postgresql.operations import DatabaseOperations

class MockConnection:
    pass

ops = DatabaseOperations(MockConnection())

sql, params = ops.date_extract_sql("WEEK_DAY", "date_column", ())
print(f"Generated SQL: {sql}")

sql, params = ops.date_extract_sql("ISO_WEEK_DAY", "date_column", ())
print(f"Generated SQL: {sql}")

sql, params = ops.date_extract_sql("ISO_YEAR", "date_column", ())
print(f"Generated SQL: {sql}")
```

Output:
```
Generated SQL: EXTRACT(WEEK_DAY FROM date_column)
Generated SQL: EXTRACT(ISO_WEEK_DAY FROM date_column)
Generated SQL: EXTRACT(ISO_YEAR FROM date_column)
```

These SQL statements will fail when executed against PostgreSQL because "WEEK_DAY", "ISO_WEEK_DAY", and "ISO_YEAR" are not valid PostgreSQL EXTRACT fields. PostgreSQL expects "DOW", "ISODOW", and "ISOYEAR" respectively.

## Why This Is A Bug

1. The function has special handling for lowercase versions ("week_day" → "DOW", "iso_week_day" → "ISODOW", "iso_year" → "ISOYEAR") but not for uppercase versions
2. The regex `[A-Z_]+` only prevents SQL injection but doesn't validate field names against PostgreSQL's allowed EXTRACT fields
3. Invalid field names like "WEEK_DAY" pass validation and generate SQL that will fail at runtime with: `ERROR: invalid extract field name "WEEK_DAY"`
4. This creates an inconsistency: lowercase special names are mapped correctly, but uppercase variants generate invalid SQL

## Fix

The function should either:
1. Reject invalid field names, OR
2. Handle uppercase variants of special cases

Here's a patch that handles uppercase variants:

```diff
--- a/django/db/backends/postgresql/operations.py
+++ b/django/db/backends/postgresql/operations.py
@@ -87,6 +87,13 @@ class DatabaseOperations(BaseDatabaseOperations):
     def date_extract_sql(self, lookup_type, sql, params):
         # https://www.postgresql.org/docs/current/functions-datetime.html#FUNCTIONS-DATETIME-EXTRACT
         if lookup_type == "week_day":
             # For consistency across backends, we return Sunday=1, Saturday=7.
             return f"EXTRACT(DOW FROM {sql}) + 1", params
         elif lookup_type == "iso_week_day":
             return f"EXTRACT(ISODOW FROM {sql})", params
         elif lookup_type == "iso_year":
             return f"EXTRACT(ISOYEAR FROM {sql})", params
+        elif lookup_type == "WEEK_DAY":
+            return f"EXTRACT(DOW FROM {sql}) + 1", params
+        elif lookup_type == "ISO_WEEK_DAY":
+            return f"EXTRACT(ISODOW FROM {sql})", params
+        elif lookup_type == "ISO_YEAR":
+            return f"EXTRACT(ISOYEAR FROM {sql})", params

         lookup_type = lookup_type.upper()
         if not self._extract_format_re.fullmatch(lookup_type):
             raise ValueError(f"Invalid lookup type: {lookup_type!r}")
         return f"EXTRACT({lookup_type} FROM {sql})", params
```

Alternatively, for stricter validation, replace the permissive regex with a whitelist:

```diff
--- a/django/db/backends/postgresql/operations.py
+++ b/django/db/backends/postgresql/operations.py
@@ -85,6 +85,14 @@ class DatabaseOperations(BaseDatabaseOperations):
     # EXTRACT format cannot be passed in parameters.
     _extract_format_re = _lazy_re_compile(r"[A-Z_]+")
+    _valid_extract_fields = frozenset([
+        'CENTURY', 'DAY', 'DECADE', 'DOW', 'DOY', 'EPOCH', 'HOUR',
+        'ISODOW', 'ISOYEAR', 'JULIAN', 'MICROSECONDS', 'MILLENNIUM',
+        'MILLISECONDS', 'MINUTE', 'MONTH', 'QUARTER', 'SECOND',
+        'TIMEZONE', 'TIMEZONE_HOUR', 'TIMEZONE_MINUTE', 'WEEK', 'YEAR'
+    ])

     def date_extract_sql(self, lookup_type, sql, params):
         # https://www.postgresql.org/docs/current/functions-datetime.html#FUNCTIONS-DATETIME-EXTRACT
         if lookup_type == "week_day":
             # For consistency across backends, we return Sunday=1, Saturday=7.
             return f"EXTRACT(DOW FROM {sql}) + 1", params
         elif lookup_type == "iso_week_day":
             return f"EXTRACT(ISODOW FROM {sql})", params
         elif lookup_type == "iso_year":
             return f"EXTRACT(ISOYEAR FROM {sql})", params

         lookup_type = lookup_type.upper()
-        if not self._extract_format_re.fullmatch(lookup_type):
+        if lookup_type not in self._valid_extract_fields:
             raise ValueError(f"Invalid lookup type: {lookup_type!r}")
         return f"EXTRACT({lookup_type} FROM {sql})", params
```