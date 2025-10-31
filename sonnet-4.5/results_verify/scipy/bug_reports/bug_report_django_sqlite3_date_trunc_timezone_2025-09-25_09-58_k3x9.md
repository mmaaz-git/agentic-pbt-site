# Bug Report: Django SQLite3 _sqlite_date_trunc TypeError with Timezone

**Target**: `django.db.backends.sqlite3._functions._sqlite_date_trunc`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `_sqlite_date_trunc` function crashes with a TypeError when processing date-only strings (without time component) in timezone-aware database connections. The function attempts to call `.replace(tzinfo=...)` on a `date` object, which only works on `datetime` objects.

## Property-Based Test

```python
@given(st.sampled_from(['year', 'month', 'day', 'week', 'quarter']))
@settings(max_examples=100)
def test_date_trunc_idempotent(self, lookup_type):
    dt = "2023-06-15"
    conn_tzname = "UTC"
    truncated_once = funcs._sqlite_date_trunc(lookup_type, dt, None, conn_tzname)
    if truncated_once is not None:
        truncated_twice = funcs._sqlite_date_trunc(lookup_type, truncated_once, None, conn_tzname)
        assert truncated_once == truncated_twice
```

**Failing input**: `dt="2023-06-15"`, `conn_tzname="UTC"`, `lookup_type="year"`

## Reproducing the Bug

```python
import django.db.backends.sqlite3._functions as funcs

dt_string = "2023-06-15"
conn_tzname = "UTC"

result = funcs._sqlite_date_trunc('year', dt_string, None, conn_tzname)
```

**Output:**
```
TypeError: replace() got an unexpected keyword argument 'tzinfo'
```

## Why This Is A Bug

The `_sqlite_datetime_parse` helper function is called by `_sqlite_date_trunc` and assumes that `typecast_timestamp()` always returns a `datetime.datetime` object. However, when the input string contains only a date (e.g., "2023-06-15"), `typecast_timestamp()` returns a `datetime.date` object instead.

When `conn_tzname` is provided (which happens in timezone-aware Django configurations with `USE_TZ=True`), the code attempts to call `dt.replace(tzinfo=...)` on line 114 of `_functions.py`. This fails because `date` objects don't have a `tzinfo` attribute or parameter.

This affects real Django usage when:
1. A DateField (not DateTimeField) is queried with date truncation
2. The database connection has timezone support enabled (`USE_TZ=True`)
3. The SQLite backend attempts to perform date truncation operations

## Fix

```diff
--- a/django/db/backends/sqlite3/_functions.py
+++ b/django/db/backends/sqlite3/_functions.py
@@ -108,10 +108,18 @@ def _sqlite_datetime_parse(dt, tzname=None, conn_tzname=None):
     if dt is None:
         return None
     try:
         dt = typecast_timestamp(dt)
     except (TypeError, ValueError):
         return None
+
+    # typecast_timestamp returns date for date-only strings
+    # Convert to datetime if we need timezone handling
+    if isinstance(dt, datetime.date) and not isinstance(dt, datetime.datetime):
+        if conn_tzname or tzname:
+            # Convert date to datetime at midnight for timezone handling
+            dt = datetime.datetime.combine(dt, datetime.time.min)
+        else:
+            return dt
+
     if conn_tzname:
         dt = dt.replace(tzinfo=zoneinfo.ZoneInfo(conn_tzname))
     if tzname is not None and tzname != conn_tzname:
```