# Bug Report: django.db.backends.mysql.operations Typo in Error Message

**Target**: `django.db.backends.mysql.operations.DatabaseOperations.date_extract_sql`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `date_extract_sql` method contains a typo in its error message: "loookup" instead of "lookup" (three o's instead of two).

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume, settings
from django.db.backends.mysql.operations import DatabaseOperations
from unittest.mock import Mock
import re

@given(st.text(alphabet=st.characters(blacklist_categories=('Cc', 'Cs')), min_size=0, max_size=50))
@settings(max_examples=1000)
def test_date_extract_sql_invalid_lookup_raises_error(lookup_type):
    assume(not re.fullmatch(r'[A-Z_]+', lookup_type))
    assume(lookup_type not in ['week_day', 'iso_week_day', 'week', 'iso_year'])

    connection = Mock()
    connection.timezone_name = 'UTC'
    ops = DatabaseOperations(connection)

    try:
        result = ops.date_extract_sql(lookup_type, 'col', [])
    except ValueError as e:
        error_msg = str(e)
        assert 'loookup' not in error_msg, f"Error message has typo: {error_msg}"
```

**Failing input**: `''` (empty string)

## Reproducing the Bug

```python
import django
from django.conf import settings

if not settings.configured:
    settings.configure(
        DATABASES={'default': {'ENGINE': 'django.db.backends.sqlite3', 'NAME': ':memory:'}},
        INSTALLED_APPS=[],
        USE_TZ=True,
    )
    django.setup()

from django.db.backends.mysql.operations import DatabaseOperations
from unittest.mock import Mock

connection = Mock()
connection.timezone_name = 'UTC'
ops = DatabaseOperations(connection)

try:
    ops.date_extract_sql('', 'col', [])
except ValueError as e:
    print(f"Error message: {e}")
```

Output:
```
Error message: Invalid loookup type: ''
```

## Why This Is A Bug

The error message contains a typo ("loookup" with three o's instead of "lookup" with two o's), which violates professional code quality standards and makes the error message harder to understand for users.

## Fix

```diff
--- a/django/db/backends/mysql/operations.py
+++ b/django/db/backends/mysql/operations.py
@@ -62,7 +62,7 @@ class DatabaseOperations(BaseDatabaseOperations):
             # EXTRACT returns 1-53 based on ISO-8601 for the week number.
             lookup_type = lookup_type.upper()
             if not self._extract_format_re.fullmatch(lookup_type):
-                raise ValueError(f"Invalid loookup type: {lookup_type!r}")
+                raise ValueError(f"Invalid lookup type: {lookup_type!r}")
             return f"EXTRACT({lookup_type} FROM {sql})", params

     def date_trunc_sql(self, lookup_type, sql, params, tzname=None):
```