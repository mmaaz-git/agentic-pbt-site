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