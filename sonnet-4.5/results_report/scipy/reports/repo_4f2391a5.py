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