# Bug Report: django.db.backends.sqlite3 quote_name Missing Quote Escaping

**Target**: `django.db.backends.sqlite3.operations.DatabaseOperations.quote_name`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `quote_name` method does not properly escape embedded double quotes in identifier names, creating malformed SQL when identifiers contain quote characters. According to SQLite's syntax, double quotes within quoted identifiers must be escaped by doubling them.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings, example
from django.db.backends.sqlite3.operations import DatabaseOperations


def is_properly_quoted(name, quoted):
    if not quoted.startswith('"') or not quoted.endswith('"'):
        return False
    inner = quoted[1:-1]
    unescaped = inner.replace('""', '"')
    return unescaped == name


@given(name=st.text(min_size=1, max_size=50))
@settings(max_examples=1000)
@example(name='foo"bar')
def test_quote_name_escapes_embedded_quotes(name):
    if name.startswith('"') and name.endswith('"'):
        return
    ops = DatabaseOperations(connection=None)
    quoted = ops.quote_name(name)
    assert is_properly_quoted(name, quoted)
```

**Failing input**: `name='foo"bar'`

## Reproducing the Bug

```python
from django.db.backends.sqlite3.operations import DatabaseOperations

ops = DatabaseOperations(connection=None)
result = ops.quote_name('foo"bar')

print(result)
```

Output: `"foo"bar"`

This creates malformed SQL. SQLite will interpret this as:
- `"foo"` (identifier)
- `bar` (unquoted, causing syntax error)
- `"` (unterminated quote)

## Why This Is A Bug

According to SQLite documentation, to include a literal double-quote character in a double-quoted identifier, it must be doubled. For example, the identifier `foo"bar` should be quoted as `"foo""bar"`.

The current implementation simply wraps the name in quotes without escaping embedded quotes:
```python
return '"%s"' % name  # foo"bar becomes "foo"bar" (wrong!)
```

This creates invalid SQL for any identifier containing double quotes. While uncommon, SQLite allows any characters in quoted identifiers, so Django should handle this correctly.

## Fix

Escape embedded double quotes by doubling them before quoting:

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

Note: This bug likely exists in other Django database backends as well (e.g., PostgreSQL uses identical logic).