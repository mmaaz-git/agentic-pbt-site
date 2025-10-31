# Bug Report: django.db.backends.mysql date_extract_sql Typo

**Target**: `django.db.backends.mysql.operations.DatabaseOperations.date_extract_sql`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The error message in `date_extract_sql` contains a typo: "Invalid loookup type" instead of "Invalid lookup type" (three 'o's instead of two).

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings as hyp_settings
from django.db.backends.mysql.operations import DatabaseOperations
from unittest.mock import Mock

@given(st.text(min_size=1, max_size=20))
@hyp_settings(max_examples=500)
def test_date_extract_sql_error_message_typo(lookup_type):
    ops = DatabaseOperations(connection=Mock())
    sql = "DATE_COLUMN"
    params = []

    try:
        ops.date_extract_sql(lookup_type, sql, params)
    except ValueError as e:
        error_msg = str(e)
        if "loookup" in error_msg:
            assert False, f"Typo found: {error_msg!r}"
```

**Failing input**: Any invalid lookup type, e.g., `"invalid$type"`

## Reproducing the Bug

```python
from django.db.backends.mysql.operations import DatabaseOperations
from unittest.mock import Mock

ops = DatabaseOperations(connection=Mock())

try:
    ops.date_extract_sql("invalid$type", "sql", [])
except ValueError as e:
    print(f"Error message: {e}")
    assert "loookup" in str(e)
    print("BUG: Typo confirmed - 'loookup' instead of 'lookup'")
```

## Why This Is A Bug

The error message is meant to be user-facing and should have correct spelling. The word "lookup" is misspelled as "loookup" with three 'o's instead of two.

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
```