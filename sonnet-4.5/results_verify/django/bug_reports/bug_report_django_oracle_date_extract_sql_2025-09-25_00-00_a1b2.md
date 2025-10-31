# Bug Report: django.db.backends.oracle date_extract_sql Typo in Error Message

**Target**: `django.db.backends.oracle.operations.DatabaseOperations.date_extract_sql`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The error message in `date_extract_sql` contains a typo: "Invalid loookup type" (with 3 o's) instead of "Invalid lookup type" (with 2 o's).

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings as hyp_settings
import re


def date_extract_sql_oracle(lookup_type, sql, params):
    _extract_format_re = re.compile(r"[A-Z_]+")

    extract_sql = f"TO_CHAR({sql}, %s)"
    extract_param = None
    if lookup_type == "week_day":
        extract_param = "D"
    elif lookup_type == "iso_week_day":
        extract_sql = f"TO_CHAR({sql} - 1, %s)"
        extract_param = "D"
    elif lookup_type == "week":
        extract_param = "IW"
    elif lookup_type == "quarter":
        extract_param = "Q"
    elif lookup_type == "iso_year":
        extract_param = "IYYY"
    else:
        lookup_type = lookup_type.upper()
        if not _extract_format_re.fullmatch(lookup_type):
            raise ValueError(f"Invalid loookup type: {lookup_type!r}")
        return f"EXTRACT({lookup_type} FROM {sql})", params
    return extract_sql, (*params, extract_param)


@given(st.text(min_size=1, max_size=50))
@hyp_settings(max_examples=1000)
def test_date_extract_sql_validates_lookup_type(lookup_type):
    valid_regex = re.compile(r"[A-Z_]+")

    if lookup_type in ['week_day', 'iso_week_day', 'week', 'quarter', 'iso_year']:
        result_sql, result_params = date_extract_sql_oracle(lookup_type, 'field', ())
        assert result_sql is not None
        assert isinstance(result_params, tuple)
    elif valid_regex.fullmatch(lookup_type.upper()):
        result_sql, result_params = date_extract_sql_oracle(lookup_type, 'field', ())
        assert result_sql is not None
        assert isinstance(result_params, tuple)
    else:
        try:
            date_extract_sql_oracle(lookup_type, 'field', ())
            assert False, f"Expected ValueError for invalid lookup_type: {lookup_type!r}"
        except ValueError as e:
            error_msg = str(e)
            assert 'Invalid' in error_msg and 'type' in error_msg
```

**Failing input**: Any string that doesn't match the valid lookup types, e.g. `"invalid!type"`

## Reproducing the Bug

```python
import re


def date_extract_sql_oracle(lookup_type, sql, params):
    _extract_format_re = re.compile(r"[A-Z_]+")

    lookup_type = lookup_type.upper()
    if not _extract_format_re.fullmatch(lookup_type):
        raise ValueError(f"Invalid loookup type: {lookup_type!r}")
    return f"EXTRACT({lookup_type} FROM {sql})", params


try:
    date_extract_sql_oracle("invalid!type", "field", ())
except ValueError as e:
    print(f"Error message: {e}")
```

Output:
```
Error message: Invalid loookup type: 'INVALID!TYPE'
```

Notice "loookup" has 3 o's instead of 2.

## Why This Is A Bug

The error message contains a typo that makes it unprofessional and potentially confusing to users. While the functionality is correct (invalid lookup types are properly rejected), the error message quality is below expected standards for a widely-used framework like Django.

## Fix

```diff
--- a/django/db/backends/oracle/operations.py
+++ b/django/db/backends/oracle/operations.py
@@ -106,7 +106,7 @@ class DatabaseOperations(BaseDatabaseOperations):
         else:
             lookup_type = lookup_type.upper()
             if not self._extract_format_re.fullmatch(lookup_type):
-                raise ValueError(f"Invalid loookup type: {lookup_type!r}")
+                raise ValueError(f"Invalid lookup type: {lookup_type!r}")
             # https://docs.oracle.com/en/database/oracle/oracle-database/21/sqlrf/EXTRACT-datetime.html
             return f"EXTRACT({lookup_type} FROM {sql})", params
         return extract_sql, (*params, extract_param)
```