# Bug Report: django.db.backends.utils.split_tzname_delta Incorrectly Formats HHMM Timezone Offsets

**Target**: `django.db.backends.utils.split_tzname_delta`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `split_tzname_delta` function incorrectly formats timezone offsets when provided in HHMM format (e.g., '0530'), producing '0530:00' instead of the expected '05:30', which creates invalid PostgreSQL timezone offset syntax.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings
from django.utils.dateparse import parse_time


def split_tzname_delta(tzname):
    """
    Split a time zone name into a 3-tuple of (name, sign, offset).
    This is a copy of the Django function for testing.
    """
    for sign in ["+", "-"]:
        if sign in tzname:
            name, offset = tzname.rsplit(sign, 1)
            if offset and parse_time(offset):
                if ":" not in offset:
                    offset = f"{offset}:00"
                return name, sign, offset
    return tzname, None, None


@given(
    st.text(min_size=1, max_size=30, alphabet='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_/'),
    st.sampled_from(['+', '-']),
    st.integers(min_value=0, max_value=23),
    st.integers(min_value=0, max_value=59)
)
@settings(max_examples=100)
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


# Run the test
if __name__ == "__main__":
    print("Running Hypothesis property-based test...")
    print("Testing split_tzname_delta function with HHMM format inputs")
    print("=" * 60)

    try:
        test_split_tzname_delta_hhmm_format_produces_invalid_offset()
        print("All tests passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")
        print("\nThis demonstrates the bug where HHMM format produces HHMM:00 instead of HH:MM")
```

<details>

<summary>
**Failing input**: `tzname='A', sign='+', hours=0, minutes=0`
</summary>
```
Running Hypothesis property-based test...
Testing split_tzname_delta function with HHMM format inputs
============================================================
Test failed: split_tzname_delta should format offset as HH:MM, not HHMM:00. Input: 'A+0000', Expected offset: '00:00', Got: '0000:00'

This demonstrates the bug where HHMM format produces HHMM:00 instead of HH:MM
```
</details>

## Reproducing the Bug

```python
import sys
import os
# Add Django path to sys.path
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

# Simple implementation to test the function
def split_tzname_delta(tzname):
    """
    Split a time zone name into a 3-tuple of (name, sign, offset).
    This is a copy of the Django function for testing.
    """
    from django.utils.dateparse import parse_time
    for sign in ["+", "-"]:
        if sign in tzname:
            name, offset = tzname.rsplit(sign, 1)
            if offset and parse_time(offset):
                if ":" not in offset:
                    offset = f"{offset}:00"
                return name, sign, offset
    return tzname, None, None


# Test case 1: HHMM format without colon
input_tz = 'UTC+0530'
name, sign, offset = split_tzname_delta(input_tz)
print(f"Test 1 - Input: {input_tz}")
print(f"Result: name={name!r}, sign={sign!r}, offset={offset!r}")
print(f"Expected offset: '05:30', Actual offset: {offset!r}")
print()

# Test case 2: Edge case with 0000
input_tz = 'A+0000'
name, sign, offset = split_tzname_delta(input_tz)
print(f"Test 2 - Input: {input_tz}")
print(f"Result: name={name!r}, sign={sign!r}, offset={offset!r}")
print(f"Expected offset: '00:00', Actual offset: {offset!r}")
print()

# Test case 3: Another HHMM case
input_tz = 'EST-0245'
name, sign, offset = split_tzname_delta(input_tz)
print(f"Test 3 - Input: {input_tz}")
print(f"Result: name={name!r}, sign={sign!r}, offset={offset!r}")
print(f"Expected offset: '02:45', Actual offset: {offset!r}")
print()

# Test case 4: Valid input with HH format (2 digits)
input_tz = 'UTC+05'
name, sign, offset = split_tzname_delta(input_tz)
print(f"Test 4 - Input: {input_tz}")
print(f"Result: name={name!r}, sign={sign!r}, offset={offset!r}")
print(f"Expected offset: '05:00', Actual offset: {offset!r}")
print()

# Test case 5: Correct format with colon works
input_tz = 'UTC+05:30'
name, sign, offset = split_tzname_delta(input_tz)
print(f"Test 5 (correct format) - Input: {input_tz}")
print(f"Result: name={name!r}, sign={sign!r}, offset={offset!r}")
print(f"This format works correctly!")
print()

# Show what PostgreSQL would receive
print("=" * 60)
print("PostgreSQL AT TIME ZONE impact:")
print("=" * 60)

def prepare_tzname_delta(tzname):
    """Mimics PostgreSQL backend's _prepare_tzname_delta."""
    name, sign, offset = split_tzname_delta(tzname)
    if offset:
        sign = "-" if sign == "+" else "+"
        return f"{name}{sign}{offset}"
    return tzname

for test_tz in ['UTC+0530', 'EST-0245', 'A+0000']:
    prepared = prepare_tzname_delta(test_tz)
    print(f"Input: {test_tz:15} -> Prepared for PostgreSQL: {prepared}")
    print(f"  Expected format: {test_tz.split('+')[0] if '+' in test_tz else test_tz.split('-')[0]}{'-' if '+' in test_tz else '+'}{test_tz[-2:] if len(test_tz.split('+' if '+' in test_tz else '-')[-1]) == 4 else test_tz[-2:]}:{test_tz[-2:] if len(test_tz.split('+' if '+' in test_tz else '-')[-1]) == 4 else '00'}")
    if '0530:00' in prepared or '0245:00' in prepared or '0000:00' in prepared:
        print(f"  ❌ INVALID: PostgreSQL expects HH:MM format, not HHMM:00")
    print()
```

<details>

<summary>
Output showing incorrect HHMM:00 format instead of HH:MM
</summary>
```
Test 1 - Input: UTC+0530
Result: name='UTC', sign='+', offset='0530:00'
Expected offset: '05:30', Actual offset: '0530:00'

Test 2 - Input: A+0000
Result: name='A', sign='+', offset='0000:00'
Expected offset: '00:00', Actual offset: '0000:00'

Test 3 - Input: EST-0245
Result: name='EST', sign='-', offset='0245:00'
Expected offset: '02:45', Actual offset: '0245:00'

Test 4 - Input: UTC+05
Result: name='UTC', sign='+', offset='05:00'
Expected offset: '05:00', Actual offset: '05:00'

Test 5 (correct format) - Input: UTC+05:30
Result: name='UTC', sign='+', offset='05:30'
This format works correctly!

============================================================
PostgreSQL AT TIME ZONE impact:
============================================================
Input: UTC+0530        -> Prepared for PostgreSQL: UTC-0530:00
  Expected format: UTC-30:30
  ❌ INVALID: PostgreSQL expects HH:MM format, not HHMM:00

Input: EST-0245        -> Prepared for PostgreSQL: EST+0245:00
  Expected format: EST+45:45
  ❌ INVALID: PostgreSQL expects HH:MM format, not HHMM:00

Input: A+0000          -> Prepared for PostgreSQL: A-0000:00
  Expected format: A-00:00
  ❌ INVALID: PostgreSQL expects HH:MM format, not HHMM:00
```
</details>

## Why This Is A Bug

This bug violates expected behavior in multiple ways:

1. **Incorrect Format Assumption**: The function assumes all offsets without colons are in HH format (hours only), but `django.utils.dateparse.parse_time()` successfully parses HHMM format. When the function receives '0530', parse_time correctly interprets it as 5 hours and 30 minutes, but the code then blindly appends ':00', producing the invalid '0530:00'.

2. **PostgreSQL Compatibility Issue**: The resulting format '0530:00' is not valid for PostgreSQL's AT TIME ZONE clause. PostgreSQL expects timezone offsets in the format `[+/-]HH[:MM[:SS]]`. The string '0530:00' would be interpreted as 530 hours and 0 minutes, which is nonsensical and would likely cause a database error.

3. **Inconsistent with ISO 8601**: HHMM format is a valid ISO 8601 timezone offset representation (e.g., '+0530' for UTC+5:30). The function should handle this standard format correctly.

4. **Silent Data Corruption**: Rather than rejecting invalid input or correctly parsing valid input, the function produces malformed output that appears valid but will cause failures downstream in database operations.

## Relevant Context

The `split_tzname_delta` function is located in `/django/db/backends/utils.py` at lines 195-206. It's used by multiple database backends:

- PostgreSQL uses it in `_prepare_tzname_delta()` and `_convert_sql_to_tz()` methods (django/db/backends/postgresql/operations.py:107-118)
- These methods generate SQL with AT TIME ZONE clauses for datetime operations
- The bug affects any Django application using timezone-aware datetime queries with PostgreSQL

The function already uses `django.utils.dateparse.parse_time()` which correctly handles:
- '05' → time(5, 0, 0)
- '0530' → time(5, 30, 0)
- '05:30' → time(5, 30, 0)

The issue is the simplistic check `if ":" not in offset` that doesn't distinguish between HH format (2 digits) and HHMM format (4 digits).

Documentation: https://docs.djangoproject.com/en/stable/ref/models/database-functions/#extract
PostgreSQL AT TIME ZONE: https://www.postgresql.org/docs/current/functions-datetime.html#FUNCTIONS-DATETIME-ZONECONVERT

## Proposed Fix

```diff
--- a/django/db/backends/utils.py
+++ b/django/db/backends/utils.py
@@ -195,12 +195,17 @@ def split_tzname_delta(tzname):
     """
     Split a time zone name into a 3-tuple of (name, sign, offset).
     """
     for sign in ["+", "-"]:
         if sign in tzname:
             name, offset = tzname.rsplit(sign, 1)
-            if offset and parse_time(offset):
+            parsed_time = parse_time(offset) if offset else None
+            if parsed_time:
                 if ":" not in offset:
-                    offset = f"{offset}:00"
+                    # Handle HHMM format by extracting hours and minutes from parsed time
+                    if len(offset) == 4 and offset.isdigit():
+                        offset = f"{offset[:2]}:{offset[2:]}"
+                    else:
+                        # HH format - append :00 for minutes
+                        offset = f"{offset}:00"
                 return name, sign, offset
     return tzname, None, None
```