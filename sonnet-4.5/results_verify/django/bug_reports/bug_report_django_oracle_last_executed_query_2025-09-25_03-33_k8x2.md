# Bug Report: django.db.backends.oracle Last Executed Query Duplicate Parameters

**Target**: `django.db.backends.oracle.operations.DatabaseOperations.last_executed_query`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When `last_executed_query()` receives a list of parameters with duplicates, it incorrectly uses `dict.fromkeys()` to deduplicate before enumeration, causing parameter index misalignment and leaving some placeholders unreplaced in the returned query string.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

from hypothesis import given, strategies as st, settings, example
from unittest.mock import Mock
from django.db.backends.oracle.operations import DatabaseOperations


@given(st.lists(st.integers(min_value=0, max_value=100), min_size=1, max_size=20))
@settings(max_examples=1000)
@example([1, 2, 1, 3])
@example([5, 5, 5])
@example([1, 1])
def test_last_executed_query_with_duplicate_params(params):
    ops = DatabaseOperations(connection=Mock())

    cursor = Mock()
    placeholders = " ".join([f":arg{i}" for i in range(len(params))])
    cursor.statement = placeholders

    result = ops.last_executed_query(cursor, "dummy_sql", params)

    for i in range(len(params)):
        placeholder = f":arg{i}"
        assert placeholder not in result, (
            f"Placeholder {placeholder} was not replaced! "
            f"Original params: {params}, Result: {result}"
        )
```

**Failing input**: `[1, 2, 1, 3]` (or any list with duplicate values)

## Reproducing the Bug

```python
from unittest.mock import Mock
from django.db.backends.oracle.operations import DatabaseOperations

ops = DatabaseOperations(connection=Mock())

cursor = Mock()
cursor.statement = "SELECT * FROM table WHERE id = :arg0 OR id = :arg1 OR id = :arg2 OR id = :arg3"

params = [1, 2, 1, 3]

result = ops.last_executed_query(cursor, "SELECT ...", params)

print("Result:", result)
print("Expected: All :arg0, :arg1, :arg2, :arg3 replaced")
print("Actual: :arg3 NOT replaced because dict.fromkeys([1, 2, 1, 3]) only has 3 unique values")
```

## Why This Is A Bug

The function attempts to substitute parameter placeholders in the SQL statement. When params is a list/tuple with duplicates, the code uses `dict.fromkeys(params)` which removes duplicates, then enumerates the deduplicated values. This causes index misalignment:

- Original params: `[1, 2, 1, 3]` (4 parameters, indices 0-3)
- After `dict.fromkeys()`: `{1: None, 2: None, 3: None}` (3 unique values)
- After `enumerate()`: `[(0, 1), (1, 2), (2, 3)]`
- Resulting mapping: `{":arg0": 1, ":arg1": 2, ":arg2": 3}`
- Missing: `":arg3"` is never mapped!

If the SQL statement contains `:arg3`, it will remain unreplaced in the returned query string, making debugging confusing as the query won't match what was actually executed.

## Fix

```diff
--- a/django/db/backends/oracle/operations.py
+++ b/django/db/backends/oracle/operations.py
@@ -338,7 +338,7 @@ class DatabaseOperations(BaseDatabaseOperations):
         if statement and params:
             if isinstance(params, (tuple, list)):
                 params = {
-                    f":arg{i}": param for i, param in enumerate(dict.fromkeys(params))
+                    f":arg{i}": param for i, param in enumerate(params)
                 }
             elif isinstance(params, dict):
                 params = {f":{key}": val for (key, val) in params.items()}
```

The `dict.fromkeys()` call appears to be unnecessary and harmful. Removing it ensures that all parameter indices are correctly mapped, even when the parameter list contains duplicate values. The subsequent replacement logic already handles sorting by key length to avoid partial replacements.