#!/usr/bin/env python3
"""
Minimal reproduction of the Django SQLite quote_name bug.
This demonstrates that identifiers with embedded double quotes
are not properly escaped, creating malformed SQL.
"""

import sqlite3

# Reproduce Django's quote_name function logic
def django_quote_name(name):
    """
    This is exactly how Django's SQLite backend quotes names
    (from django/db/backends/sqlite3/operations.py lines 199-202)
    """
    if name.startswith('"') and name.endswith('"'):
        return name  # Quoting once is enough.
    return '"%s"' % name

# Create a test identifier with an embedded double quote
test_name = 'foo"bar'

# Use the Django logic
quoted = django_quote_name(test_name)

print("Input identifier:", test_name)
print("Django's quoted output:", quoted)
print()

# Try to use this in actual SQLite to show it creates invalid SQL
conn = sqlite3.connect(':memory:')
cursor = conn.cursor()

# First, show what the correct quoting should be
correct_quoted = '"foo""bar"'  # Double quotes must be doubled
print("Correct SQL quoting:", correct_quoted)

# Create a table with the correctly quoted identifier
try:
    sql = f"CREATE TABLE test ({correct_quoted} INTEGER)"
    cursor.execute(sql)
    print(f"✓ Successfully created table with column: {correct_quoted}")
except sqlite3.OperationalError as e:
    print(f"✗ Failed with correctly quoted name: {e}")

# Now try with Django's incorrect quoting
print()
print("Testing Django's quoted output in SQLite:")
try:
    # This will fail because Django's output is malformed
    sql = f"CREATE TABLE test2 ({quoted} INTEGER)"
    print(f"Attempting to execute: CREATE TABLE test2 ({quoted} INTEGER)")
    cursor.execute(sql)
    print(f"✓ Successfully created table with column: {quoted}")
except sqlite3.OperationalError as e:
    print(f"✗ Failed with Django's quoting: {e}")
    print()
    print("This is a syntax error because SQLite interprets the malformed SQL as:")
    print('  "foo" (identifier) followed by bar" (unexpected token)')

conn.close()