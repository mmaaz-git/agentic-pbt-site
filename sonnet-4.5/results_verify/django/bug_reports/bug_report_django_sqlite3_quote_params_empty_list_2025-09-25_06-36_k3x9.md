# Bug Report: django.db.backends.sqlite3 _quote_params_for_last_executed_query Empty List

**Target**: `django.db.backends.sqlite3.operations.DatabaseOperations._quote_params_for_last_executed_query`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `_quote_params_for_last_executed_query` method crashes with `sqlite3.OperationalError: incomplete input` when called with an empty parameter tuple or list.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from django.db import connection

@given(st.lists(st.integers(), max_size=10))
def test_quote_params_preserves_length(params):
    connection.ensure_connection()
    result = connection.ops._quote_params_for_last_executed_query(tuple(params))
    assert len(result) == len(params)
```

**Failing input**: `params=[]`

## Reproducing the Bug

```python
import sqlite3

conn = sqlite3.connect(":memory:")
cursor = conn.cursor()

params = ()
sql = "SELECT " + ", ".join(["QUOTE(?)"] * len(params))

cursor.execute(sql, params).fetchone()
```

**Output**:
```
sqlite3.OperationalError: incomplete input
```

## Why This Is A Bug

The `_quote_params_for_last_executed_query` method is a utility function meant to quote parameters for debugging/logging purposes (as noted in the docstring). While the calling code in `last_executed_query` checks `if params:` before calling this method, making the empty list case unlikely in normal usage, the function should still handle empty parameter lists gracefully for the following reasons:

1. **Defensive programming**: Internal utility functions should not crash on edge cases
2. **Future-proofing**: If other code paths call this method directly, they may not know to avoid empty lists
3. **Consistency**: The function claims to quote parameters for display, and an empty parameter list is a valid case to display (as an empty result)

The bug occurs because when `len(params) == 0`, the code generates `sql = "SELECT "` (line 171 in operations.py), which is incomplete SQL that SQLite cannot execute.

## Fix

```diff
--- a/django/db/backends/sqlite3/operations.py
+++ b/django/db/backends/sqlite3/operations.py
@@ -154,6 +154,9 @@ class DatabaseOperations(BaseDatabaseOperations):
     def _quote_params_for_last_executed_query(self, params):
         """
         Only for last_executed_query! Don't use this to execute SQL queries!
         """
+        if not params:
+            return ()
+
         # This function is limited both by SQLITE_LIMIT_VARIABLE_NUMBER (the
         # number of parameters, default = 999) and SQLITE_MAX_COLUMN (the
         # number of return values, default = 2000). Since Python's sqlite3
```