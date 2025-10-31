#!/usr/bin/env python3
"""Minimal reproduction of Django MySQL date_extract_sql typo bug"""

from django.db.backends.mysql.operations import DatabaseOperations
from unittest.mock import Mock

# Create a DatabaseOperations instance with a mock connection
ops = DatabaseOperations(connection=Mock())

# Try to call date_extract_sql with an invalid lookup type
# This should trigger the ValueError with the typo
try:
    ops.date_extract_sql("invalid$type", "DATE_COLUMN", [])
except ValueError as e:
    print(f"Error message: {e}")
    print(f"\nChecking for typo...")
    if "loookup" in str(e):
        print("BUG CONFIRMED: Error message contains 'loookup' instead of 'lookup'")
        print("The typo has three 'o's instead of two.")
    else:
        print("No typo found - bug may have been fixed")