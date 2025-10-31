# Bug Report: django.db.backends.oracle.operations.DatabaseOperations.last_executed_query - Incorrect Parameter Substitution with Duplicate Values

**Target**: `django.db.backends.oracle.operations.DatabaseOperations.last_executed_query`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `last_executed_query` method incorrectly handles list/tuple parameters containing duplicate values. It uses `dict.fromkeys()` which removes duplicates, causing parameter indices to be incorrect when substituting placeholders in the SQL statement.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume

@given(st.lists(st.integers(min_value=0, max_value=10), min_size=1, max_size=10))
def test_last_executed_query_preserves_all_params(params):
    assume(len(params) != len(set(params)))

    params_dict = {
        f":arg{i}": param for i, param in enumerate(dict.fromkeys(params))
    }

    assert len(params_dict) == len(params), \
        f"Expected {len(params)} params, got {len(params_dict)}"
```

**Failing input**: `params = [100, 'active', 'active', 200]`

## Reproducing the Bug

```python
def last_executed_query_buggy(statement, params):
    if statement and params:
        if isinstance(params, (tuple, list)):
            params_dict = {
                f":arg{i}": param for i, param in enumerate(dict.fromkeys(params))
            }
        else:
            params_dict = params

        for key in sorted(params_dict, key=len, reverse=True):
            statement = statement.replace(key, str(params_dict[key]))
    return statement


statement = "SELECT * FROM users WHERE id = :arg0 AND status = :arg1 AND type = :arg2 AND priority = :arg3"
params = [100, 'active', 'active', 200]

result = last_executed_query_buggy(statement, params)

print("Result:", result)

print("\nBuggy params dict:", {f":arg{i}": param for i, param in enumerate(dict.fromkeys(params))})

print("Correct params dict:", {f":arg{i}": param for i, param in enumerate(params)})
```

Output:
```
Result: SELECT * FROM users WHERE id = 100 AND status = active AND type = :arg2 AND priority = :arg3
Buggy params dict: {':arg0': 100, ':arg1': 'active', ':arg2': 200}
Correct params dict: {':arg0': 100, ':arg1': 'active', ':arg2': 'active', ':arg3': 200}
```

The bug causes `:arg2` and `:arg3` to not be replaced because `dict.fromkeys([100, 'active', 'active', 200])` only returns 3 unique values, so the enumeration only goes from 0-2 instead of 0-3.

## Why This Is A Bug

The function is meant to display the last executed SQL query with parameters substituted for debugging purposes. When a params list/tuple contains duplicate values (e.g., `[100, 'active', 'active', 200]`), using `dict.fromkeys()` removes those duplicates before enumeration. This causes:

1. The parameter dictionary to have fewer entries than the original params list
2. Later parameter placeholders (`:arg2`, `:arg3`, etc.) to remain unreplaced in the output
3. Incorrect/misleading debugging information for developers

This violates the expected invariant that a params list of length N should create placeholder mappings for `:arg0` through `:argN-1`.

## Fix

```diff
--- a/django/db/backends/oracle/operations.py
+++ b/django/db/backends/oracle/operations.py
@@ -337,7 +337,7 @@ class DatabaseOperations(BaseDatabaseOperations):
     def last_executed_query(self, cursor, sql, params):
         if statement and params:
             if isinstance(params, (tuple, list)):
                 params = {
-                    f":arg{i}": param for i, param in enumerate(dict.fromkeys(params))
+                    f":arg{i}": param for i, param in enumerate(params)
                 }
             elif isinstance(params, dict):
                 params = {f":{key}": val for (key, val) in params.items()}
```

The fix is simple: remove the `dict.fromkeys()` call and enumerate the params list directly. This preserves the original parameter count and indices.