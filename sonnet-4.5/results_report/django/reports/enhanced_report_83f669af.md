# Bug Report: django.db.backends.sqlite3.operations.DatabaseOperations.quote_name Missing Double Quote Escaping

**Target**: `django.db.backends.sqlite3.operations.DatabaseOperations.quote_name`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `quote_name` method in Django's SQLite backend fails to escape embedded double quotes in identifier names, producing syntactically invalid SQL that causes runtime errors when identifiers contain double quote characters.

## Property-Based Test

```python
#!/usr/bin/env python3
"""
Property-based test for Django's SQLite quote_name function.
This test uses Hypothesis to find inputs where the function fails to
properly escape embedded double quotes.
"""

from hypothesis import given, strategies as st, settings, example
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


def is_valid_sqlite_quoted_identifier(name, quoted):
    """
    Check if a quoted identifier is valid SQLite syntax by actually trying to use it.
    This is the most reliable way to test if the quoting is correct.
    """
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    try:
        # Try to create a table with this identifier as a column name
        sql = f"CREATE TABLE test ({quoted} INTEGER)"
        cursor.execute(sql)

        # If we get here, the SQL was valid
        # Now verify we can query with the original name (properly escaped)
        correct_quoted = '"' + name.replace('"', '""') + '"'
        cursor.execute(f"SELECT {correct_quoted} FROM test LIMIT 0")

        conn.close()
        return True
    except sqlite3.OperationalError:
        conn.close()
        return False


@given(name=st.text(min_size=1, max_size=50))
@settings(max_examples=1000)
@example(name='foo"bar')
def test_quote_name_produces_valid_sql(name):
    """Test that quote_name produces valid SQLite SQL."""
    # Skip already-quoted names (the function has special handling for these)
    if name.startswith('"') and name.endswith('"'):
        return

    # Skip empty names
    if not name.strip():
        return

    # Apply Django's quote_name logic
    quoted = django_quote_name(name)

    # Check if the quoted identifier is valid SQLite SQL
    assert is_valid_sqlite_quoted_identifier(name, quoted), (
        f"Failed for name={name!r}: "
        f"Django produced {quoted!r} which is invalid SQLite SQL. "
        f"Should be: \"{name.replace('\"', '\"\"')}\""
    )


if __name__ == "__main__":
    # Run the test
    test_quote_name_produces_valid_sql()
```

<details>

<summary>
**Failing input**: `name='foo"bar'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/37/hypo.py", line 72, in <module>
    test_quote_name_produces_valid_sql()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/37/hypo.py", line 47, in test_quote_name_produces_valid_sql
    @settings(max_examples=1000)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2062, in wrapped_test
    _raise_to_user(errors, state.settings, [], " in explicit examples")
    ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1613, in _raise_to_user
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/37/hypo.py", line 63, in test_quote_name_produces_valid_sql
    assert is_valid_sqlite_quoted_identifier(name, quoted), (
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^
AssertionError: Failed for name='foo"bar': Django produced '"foo"bar"' which is invalid SQLite SQL. Should be: "foo""bar"
Falsifying explicit example: test_quote_name_produces_valid_sql(
    name='foo"bar',
)
```
</details>

## Reproducing the Bug

```python
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
```

<details>

<summary>
SQLite syntax error when Django's quoted identifier is used
</summary>
```
Input identifier: foo"bar
Django's quoted output: "foo"bar"

Correct SQL quoting: "foo""bar"
✓ Successfully created table with column: "foo""bar"

Testing Django's quoted output in SQLite:
Attempting to execute: CREATE TABLE test2 ("foo"bar" INTEGER)
✗ Failed with Django's quoting: unrecognized token: "" INTEGER)"

This is a syntax error because SQLite interprets the malformed SQL as:
  "foo" (identifier) followed by bar" (unexpected token)
```
</details>

## Why This Is A Bug

According to the SQL standard and SQLite documentation, double quotes within quoted identifiers must be escaped by doubling them. SQLite's official documentation states: "If a double-quote character appears within a double-quoted identifier, it is escaped by doubling it."

Django's current implementation violates this requirement:
- Input: `foo"bar`
- Django produces: `"foo"bar"` (malformed)
- Should produce: `"foo""bar"` (correctly escaped)

When SQLite encounters `"foo"bar"`, it parses this as:
1. `"foo"` - a valid quoted identifier
2. `bar"` - an unexpected token, causing a syntax error

This bug causes Django's ORM to generate invalid SQL that will crash with a syntax error whenever:
- A model field name contains a double quote
- A table name contains a double quote
- Any identifier passed to `quote_name` contains a double quote

While such identifiers are rare in practice, they are valid in SQLite, and Django should handle them correctly.

## Relevant Context

This bug exists in multiple Django database backends:
- SQLite: `/django/db/backends/sqlite3/operations.py` lines 199-202
- PostgreSQL: `/django/db/backends/postgresql/operations.py` lines 196-199

Both use identical logic that fails to escape embedded quotes. This is a longstanding issue that violates SQL standards but has low practical impact due to the rarity of identifiers containing double quotes.

SQLite documentation on identifier quoting: https://www.sqlite.org/lang_keywords.html
PostgreSQL documentation on identifier quoting: https://www.postgresql.org/docs/current/sql-syntax-lexical.html#SQL-SYNTAX-IDENTIFIERS

## Proposed Fix

```diff
--- a/django/db/backends/sqlite3/operations.py
+++ b/django/db/backends/sqlite3/operations.py
@@ -199,7 +199,7 @@ class DatabaseOperations(BaseDatabaseOperations):
     def quote_name(self, name):
         if name.startswith('"') and name.endswith('"'):
             return name  # Quoting once is enough.
-        return '"%s"' % name
+        return '"%s"' % name.replace('"', '""')
```

The same fix should be applied to PostgreSQL and any other backends with identical logic.