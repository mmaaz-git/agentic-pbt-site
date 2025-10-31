# Bug Report: Django Oracle Backend last_executed_query Incorrectly Handles Duplicate Parameters

**Target**: `django.db.backends.oracle.operations.DatabaseOperations.last_executed_query`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `last_executed_query` method incorrectly deduplicates parameters using `dict.fromkeys()`, causing parameter index misalignment that leaves some placeholders unreplaced when the parameter list contains duplicate values.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings, example
from unittest.mock import Mock

def force_str(s, errors="strict"):
    """Simple force_str replacement for testing"""
    if isinstance(s, str):
        return s
    return str(s)

# This is the exact function from django/db/backends/oracle/operations.py
def last_executed_query(cursor, sql, params):
    # https://python-oracledb.readthedocs.io/en/latest/api_manual/cursor.html#Cursor.statement
    # The DB API definition does not define this attribute.
    statement = cursor.statement
    # Unlike Psycopg's `query` and MySQLdb`'s `_executed`, oracledb's
    # `statement` doesn't contain the query parameters. Substitute
    # parameters manually.
    if statement and params:
        if isinstance(params, (tuple, list)):
            params = {
                f":arg{i}": param for i, param in enumerate(dict.fromkeys(params))
            }
        elif isinstance(params, dict):
            params = {f":{key}": val for (key, val) in params.items()}
        for key in sorted(params, key=len, reverse=True):
            statement = statement.replace(
                key, force_str(params[key], errors="replace")
            )
    return statement


@given(st.lists(st.integers(min_value=0, max_value=100), min_size=1, max_size=20))
@settings(max_examples=1000)
@example([1, 2, 1, 3])
@example([5, 5, 5])
@example([1, 1])
def test_last_executed_query_with_duplicate_params(params):
    cursor = Mock()
    placeholders = " ".join([f":arg{i}" for i in range(len(params))])
    cursor.statement = placeholders

    result = last_executed_query(cursor, "dummy_sql", params)

    for i in range(len(params)):
        placeholder = f":arg{i}"
        assert placeholder not in result, (
            f"Placeholder {placeholder} was not replaced! "
            f"Original params: {params}, Result: {result}"
        )


if __name__ == "__main__":
    # Run the test
    test_last_executed_query_with_duplicate_params()
```

<details>

<summary>
**Failing input**: `[1, 2, 1, 3]`
</summary>
```
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/8/hypo.py", line 54, in <module>
  |     test_last_executed_query_with_duplicate_params()
  |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  |   File "/home/npc/pbt/agentic-pbt/worker_/8/hypo.py", line 33, in test_last_executed_query_with_duplicate_params
  |     @settings(max_examples=1000)
  |                    ^^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2062, in wrapped_test
  |     _raise_to_user(errors, state.settings, [], " in explicit examples")
  |     ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1613, in _raise_to_user
  |     raise the_error_hypothesis_found
  | ExceptionGroup: Hypothesis found 3 distinct failures in explicit examples. (3 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/8/hypo.py", line 46, in test_last_executed_query_with_duplicate_params
    |     assert placeholder not in result, (
    |            ^^^^^^^^^^^^^^^^^^^^^^^^^
    | AssertionError: Placeholder :arg3 was not replaced! Original params: [1, 2, 1, 3], Result: 1 2 3 :arg3
    | Falsifying explicit example: test_last_executed_query_with_duplicate_params(
    |     params=[1, 2, 1, 3],
    | )
    +---------------- 2 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/8/hypo.py", line 46, in test_last_executed_query_with_duplicate_params
    |     assert placeholder not in result, (
    |            ^^^^^^^^^^^^^^^^^^^^^^^^^
    | AssertionError: Placeholder :arg1 was not replaced! Original params: [5, 5, 5], Result: 5 :arg1 :arg2
    | Falsifying explicit example: test_last_executed_query_with_duplicate_params(
    |     params=[5, 5, 5],
    | )
    +---------------- 3 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/8/hypo.py", line 46, in test_last_executed_query_with_duplicate_params
    |     assert placeholder not in result, (
    |            ^^^^^^^^^^^^^^^^^^^^^^^^^
    | AssertionError: Placeholder :arg1 was not replaced! Original params: [1, 1], Result: 1 :arg1
    | Falsifying explicit example: test_last_executed_query_with_duplicate_params(
    |     params=[1, 1],
    | )
    +------------------------------------
```
</details>

## Reproducing the Bug

```python
from unittest.mock import Mock

def force_str(s, errors="strict"):
    """Simple force_str replacement for testing"""
    if isinstance(s, str):
        return s
    return str(s)

# This is the exact function from django/db/backends/oracle/operations.py
def last_executed_query(cursor, sql, params):
    # https://python-oracledb.readthedocs.io/en/latest/api_manual/cursor.html#Cursor.statement
    # The DB API definition does not define this attribute.
    statement = cursor.statement
    # Unlike Psycopg's `query` and MySQLdb`'s `_executed`, oracledb's
    # `statement` doesn't contain the query parameters. Substitute
    # parameters manually.
    if statement and params:
        if isinstance(params, (tuple, list)):
            params = {
                f":arg{i}": param for i, param in enumerate(dict.fromkeys(params))
            }
        elif isinstance(params, dict):
            params = {f":{key}": val for (key, val) in params.items()}
        for key in sorted(params, key=len, reverse=True):
            statement = statement.replace(
                key, force_str(params[key], errors="replace")
            )
    return statement

# Create a mock cursor with a statement that has 4 placeholders
cursor = Mock()
cursor.statement = "SELECT * FROM table WHERE id = :arg0 OR id = :arg1 OR id = :arg2 OR id = :arg3"

# Create a list with duplicate values - 4 parameters but only 3 unique values
params = [1, 2, 1, 3]

# Call the function that has the bug
result = last_executed_query(cursor, "SELECT * FROM table WHERE id = ? OR id = ? OR id = ? OR id = ?", params)

print("Input params:", params)
print("Number of params:", len(params))
print("Number of unique params:", len(set(params)))
print()
print("Original statement with placeholders:")
print(cursor.statement)
print()
print("Result after parameter substitution:")
print(result)
print()
print("Bug demonstration:")
if ":arg3" in result:
    print("❌ BUG: :arg3 was NOT replaced!")
    print("   This happens because dict.fromkeys([1, 2, 1, 3]) removes duplicates,")
    print("   leaving only 3 unique values, so enumeration only goes from 0-2,")
    print("   and :arg3 never gets a replacement value.")
else:
    print("✓ All placeholders were replaced correctly")
```

<details>

<summary>
AssertionError: Placeholder :arg3 was not replaced
</summary>
```
Input params: [1, 2, 1, 3]
Number of params: 4
Number of unique params: 3

Original statement with placeholders:
SELECT * FROM table WHERE id = :arg0 OR id = :arg1 OR id = :arg2 OR id = :arg3

Result after parameter substitution:
SELECT * FROM table WHERE id = 1 OR id = 2 OR id = 3 OR id = :arg3

Bug demonstration:
❌ BUG: :arg3 was NOT replaced!
   This happens because dict.fromkeys([1, 2, 1, 3]) removes duplicates,
   leaving only 3 unique values, so enumeration only goes from 0-2,
   and :arg3 never gets a replacement value.
```
</details>

## Why This Is A Bug

The `last_executed_query` method is designed to reconstruct the executed SQL query for debugging purposes by replacing numbered placeholders (`:arg0`, `:arg1`, etc.) with the actual parameter values. However, the function incorrectly uses `dict.fromkeys(params)` which removes duplicate values from the parameter list before enumeration.

This violates the expected behavior because:
1. **Parameter position matters**: In SQL, each parameter placeholder corresponds to a specific position in the parameter list, regardless of whether values are duplicated
2. **Index misalignment**: When duplicates are removed, the enumeration produces fewer indices than there are placeholders, leaving high-numbered placeholders unreplaced
3. **Debugging confusion**: The returned query string doesn't accurately represent what was executed, defeating the purpose of this debugging method

For example, with params `[1, 2, 1, 3]`:
- Expected mapping: `{":arg0": 1, ":arg1": 2, ":arg2": 1, ":arg3": 3}`
- Actual mapping due to bug: `{":arg0": 1, ":arg1": 2, ":arg2": 3}` (`:arg3` is missing!)

## Relevant Context

This bug affects Django's Oracle database backend when developers use the `connection.queries` debugging feature or when errors are logged. The Oracle backend differs from other Django database backends (PostgreSQL, MySQL) in that the cursor's statement attribute doesn't include the actual parameter values, requiring manual substitution.

The `dict.fromkeys()` call appears to be an unnecessary optimization or a misunderstanding of the requirement. The function already handles replacement correctly by sorting keys by length (line 345 in the original) to avoid partial replacements, so deduplication serves no purpose and actively breaks the functionality.

Relevant Django documentation:
- [Django Database instrumentation](https://docs.djangoproject.com/en/stable/topics/db/instrumentation/)
- [Oracle backend specifics](https://docs.djangoproject.com/en/stable/ref/databases/#oracle-notes)

Source code location: `/django/db/backends/oracle/operations.py:341`

## Proposed Fix

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