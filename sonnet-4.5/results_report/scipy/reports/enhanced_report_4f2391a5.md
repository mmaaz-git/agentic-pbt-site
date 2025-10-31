# Bug Report: Django SQLite3 _sqlite_date_trunc TypeError with Timezone-Aware Date Strings

**Target**: `django.db.backends.sqlite3._functions._sqlite_date_trunc`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `_sqlite_date_trunc` function crashes with a TypeError when processing date-only strings (e.g., "2023-06-15") in timezone-aware database connections, as it attempts to call `.replace(tzinfo=...)` on a `datetime.date` object instead of a `datetime.datetime` object.

## Property-Based Test

```python
#!/usr/bin/env python3
"""Hypothesis test for Django SQLite3 _sqlite_date_trunc bug"""

from hypothesis import given, settings
import hypothesis.strategies as st
import django.db.backends.sqlite3._functions as funcs

@given(st.sampled_from(['year', 'month', 'day', 'week', 'quarter']))
@settings(max_examples=100)
def test_date_trunc_idempotent(lookup_type):
    dt = "2023-06-15"
    conn_tzname = "UTC"
    truncated_once = funcs._sqlite_date_trunc(lookup_type, dt, None, conn_tzname)
    if truncated_once is not None:
        truncated_twice = funcs._sqlite_date_trunc(lookup_type, truncated_once, None, conn_tzname)
        assert truncated_once == truncated_twice

if __name__ == "__main__":
    # Run the test
    test_date_trunc_idempotent()
```

<details>

<summary>
**Failing input**: `lookup_type='year'`
</summary>
```
Falsifying example: test_date_trunc_idempotent(
    lookup_type='year',
)
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/7/hypo.py", line 20, in <module>
    test_date_trunc_idempotent()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/7/hypo.py", line 9, in test_date_trunc_idempotent
    @settings(max_examples=100)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/7/hypo.py", line 13, in test_date_trunc_idempotent
    truncated_once = funcs._sqlite_date_trunc(lookup_type, dt, None, conn_tzname)
  File "/home/npc/miniconda/lib/python3.13/site-packages/django/db/backends/sqlite3/_functions.py", line 129, in _sqlite_date_trunc
    dt = _sqlite_datetime_parse(dt, tzname, conn_tzname)
  File "/home/npc/miniconda/lib/python3.13/site-packages/django/db/backends/sqlite3/_functions.py", line 114, in _sqlite_datetime_parse
    dt = dt.replace(tzinfo=zoneinfo.ZoneInfo(conn_tzname))
TypeError: replace() got an unexpected keyword argument 'tzinfo'
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""Minimal reproduction case for Django SQLite3 _sqlite_date_trunc bug"""

import django.db.backends.sqlite3._functions as funcs

# Test with date-only string and timezone
dt_string = "2023-06-15"
conn_tzname = "UTC"
lookup_type = "year"

print(f"Input parameters:")
print(f"  dt_string: {dt_string}")
print(f"  conn_tzname: {conn_tzname}")
print(f"  lookup_type: {lookup_type}")
print()

try:
    result = funcs._sqlite_date_trunc(lookup_type, dt_string, None, conn_tzname)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")
    import traceback
    print("\nFull traceback:")
    traceback.print_exc()
```

<details>

<summary>
TypeError: replace() got an unexpected keyword argument 'tzinfo'
</summary>
```
Input parameters:
  dt_string: 2023-06-15
  conn_tzname: UTC
  lookup_type: year

Error occurred: TypeError: replace() got an unexpected keyword argument 'tzinfo'

Full traceback:
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/7/repo.py", line 18, in <module>
    result = funcs._sqlite_date_trunc(lookup_type, dt_string, None, conn_tzname)
  File "/home/npc/miniconda/lib/python3.13/site-packages/django/db/backends/sqlite3/_functions.py", line 129, in _sqlite_date_trunc
    dt = _sqlite_datetime_parse(dt, tzname, conn_tzname)
  File "/home/npc/miniconda/lib/python3.13/site-packages/django/db/backends/sqlite3/_functions.py", line 114, in _sqlite_datetime_parse
    dt = dt.replace(tzinfo=zoneinfo.ZoneInfo(conn_tzname))
TypeError: replace() got an unexpected keyword argument 'tzinfo'
```
</details>

## Why This Is A Bug

This bug violates Django's documented behavior and causes crashes in common usage patterns:

1. **Django's public Trunc function is documented to work with DateField**: The Django documentation explicitly states that date truncation functions should work with DateFields, TimeFields, and DateTimeFields. When using DateFields with only date-related truncation kinds (year, quarter, month, week, day), the operation should succeed.

2. **Type mismatch in internal implementation**: The `_sqlite_datetime_parse` helper function at line 110 calls `typecast_timestamp(dt)`. When the input is a date-only string like "2023-06-15", this function returns a `datetime.date` object rather than a `datetime.datetime` object. However, the code at line 114 assumes it always receives a `datetime.datetime` object and attempts to call `dt.replace(tzinfo=zoneinfo.ZoneInfo(conn_tzname))`.

3. **The `date` class lacks the required interface**: Python's `datetime.date` objects don't have a `replace()` method that accepts a `tzinfo` parameter. Only `datetime.datetime` objects have this capability. This causes the TypeError.

4. **Affects production Django configurations**: This bug manifests when:
   - Using Django's recommended `USE_TZ=True` setting for timezone-aware applications
   - Using SQLite as the database backend (Django's default database)
   - Performing date truncation queries on DateField columns
   - The database connection has timezone support enabled

5. **Breaks core ORM functionality**: This prevents users from using Django's ORM date truncation features with DateFields in timezone-aware SQLite applications, breaking queries like `DateField.objects.annotate(year=Trunc('date_field', 'year'))`.

## Relevant Context

The `_sqlite_date_trunc` function is registered as a user-defined SQL function in SQLite during connection setup (line 46 of _functions.py):
```python
create_deterministic_function("django_date_trunc", 4, _sqlite_date_trunc)
```

This function emulates PostgreSQL's DATE_TRUNC functionality for SQLite, which lacks native date truncation support. It's called internally by Django's ORM when executing date truncation queries.

The root issue is in the `_sqlite_datetime_parse` function (lines 106-125), which serves as a common parser for multiple SQLite date/time functions. The function needs to handle both `datetime.date` and `datetime.datetime` objects when timezone operations are required.

Related Django source code:
- Function definition: `/django/db/backends/sqlite3/_functions.py:128-144` (_sqlite_date_trunc)
- Parser function: `/django/db/backends/sqlite3/_functions.py:106-125` (_sqlite_datetime_parse)
- Type casting: `/django/db/backends/utils.py` (typecast_timestamp function)

## Proposed Fix

```diff
--- a/django/db/backends/sqlite3/_functions.py
+++ b/django/db/backends/sqlite3/_functions.py
@@ -1,6 +1,7 @@
 """
 Implementations of SQL functions for SQLite.
 """
+import datetime

 import functools
 import random
@@ -106,10 +107,19 @@ def _sqlite_datetime_parse(dt, tzname=None, conn_tzname=None):
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
+            # Return date object as-is when no timezone handling needed
+            return dt
+
     if conn_tzname:
         dt = dt.replace(tzinfo=zoneinfo.ZoneInfo(conn_tzname))
     if tzname is not None and tzname != conn_tzname:
```