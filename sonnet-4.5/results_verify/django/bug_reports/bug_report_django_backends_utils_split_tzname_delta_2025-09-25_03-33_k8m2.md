# Bug Report: split_tzname_delta Incorrectly Formats HHMM Timezone Offsets

**Target**: `django.db.backends.utils.split_tzname_delta`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `split_tzname_delta` function in Django's database backends incorrectly formats timezone offsets when provided in HHMM format (e.g., '0530'). Instead of converting '0530' to '05:30', it appends ':00' to produce '0530:00', which is not a valid offset format for PostgreSQL's AT TIME ZONE clause.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from django.db.backends.utils import split_tzname_delta


@given(
    st.text(min_size=1, max_size=30, alphabet='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_/'),
    st.sampled_from(['+', '-']),
    st.integers(min_value=0, max_value=23),
    st.integers(min_value=0, max_value=59)
)
def test_split_tzname_delta_hhmm_format_produces_invalid_offset(tzname, sign, hours, minutes):
    offset_hhmm = f"{hours:02d}{minutes:02d}"
    input_tz = f"{tzname}{sign}{offset_hhmm}"
    name, parsed_sign, parsed_offset = split_tzname_delta(input_tz)

    if parsed_offset is not None:
        expected_offset = f"{hours:02d}:{minutes:02d}"
        assert parsed_offset == expected_offset, (
            f"split_tzname_delta should format offset as HH:MM, not HHMM:00. "
            f"Input: {input_tz!r}, Expected offset: {expected_offset!r}, Got: {parsed_offset!r}"
        )
```

**Failing input**: `tzname='A', sign='+', hours=0, minutes=0` producing input `'A+0000'`

## Reproducing the Bug

```python
from django.db.backends.utils import split_tzname_delta
from django.db.backends.postgresql.operations import DatabaseOperations


class MockConnection:
    pass


input_tz = 'UTC+0530'
name, sign, offset = split_tzname_delta(input_tz)
print(f"Input: {input_tz}")
print(f"Result: name={name!r}, sign={sign!r}, offset={offset!r}")
print(f"Expected offset: '05:30', Actual offset: {offset!r}")

ops = DatabaseOperations(MockConnection())
prepared = ops._prepare_tzname_delta(input_tz)
print(f"\nAfter _prepare_tzname_delta: {prepared!r}")
print(f"Expected: 'UTC-05:30', Actual: {prepared!r}")
```

Output:
```
Input: UTC+0530
Result: name='UTC', sign='+', offset='0530:00'
Expected offset: '05:30', Actual offset: '0530:00'

After _prepare_tzname_delta: 'UTC-0530:00'
Expected: 'UTC-05:30', Actual: 'UTC-0530:00'
```

## Why This Is A Bug

The `split_tzname_delta` function incorrectly handles timezone offsets in HHMM format. When the offset is '0530':

1. `parse_time('0530')` successfully parses it as time(5, 30)
2. The code checks if ':' is in offset ('0530' has no colon)
3. It blindly appends ':00', producing '0530:00'

However, '0530:00' is not a valid timezone offset format. PostgreSQL expects offsets in the format `[+/-]HH[:MM[:SS]]`, where each component is properly delimited. The string '0530:00' would be interpreted incorrectly or rejected.

This affects PostgreSQL backend's `_prepare_tzname_delta` and `_convert_sql_to_tz` methods, which use `split_tzname_delta` to process timezone names for `AT TIME ZONE` clauses in datetime queries.

## Fix

The function should properly parse the HHMM format and extract hours and minutes before formatting:

```diff
diff --git a/django/db/backends/utils.py b/django/db/backends/utils.py
index 1234567..abcdefg 100644
--- a/django/db/backends/utils.py
+++ b/django/db/backends/utils.py
@@ -198,10 +198,15 @@ def split_tzname_delta(tzname):
     """
     for sign in ["+", "-"]:
         if sign in tzname:
             name, offset = tzname.rsplit(sign, 1)
-            if offset and parse_time(offset):
+            parsed_time = parse_time(offset) if offset else None
+            if parsed_time:
                 if ":" not in offset:
-                    offset = f"{offset}:00"
+                    # Handle HHMM format by extracting hours and minutes
+                    if len(offset) == 4 and offset.isdigit():
+                        offset = f"{offset[:2]}:{offset[2:]}"
+                    else:
+                        offset = f"{parsed_time.hour:02d}:{parsed_time.minute:02d}"
                 return name, sign, offset
     return tzname, None, None
```