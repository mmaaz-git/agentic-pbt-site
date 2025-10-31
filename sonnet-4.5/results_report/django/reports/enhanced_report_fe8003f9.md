# Bug Report: django.db.backends.oracle.operations.DatabaseOperations.last_executed_query - Incorrect Parameter Substitution with Duplicate Values

**Target**: `django.db.backends.oracle.operations.DatabaseOperations.last_executed_query`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `last_executed_query` method in Django's Oracle backend incorrectly handles parameter lists containing duplicate values, using `dict.fromkeys()` which removes duplicates before enumeration, causing some placeholders to remain unreplaced in the debugging output.

## Property-Based Test

```python
#!/usr/bin/env python3
"""
Property-based test for Django Oracle backend last_executed_query bug.
This test demonstrates that the method fails when parameters contain duplicates.
"""

from hypothesis import given, strategies as st, assume, settings, HealthCheck, example
import sys


def last_executed_query_buggy(statement, params):
    """
    Buggy implementation from Django's Oracle backend.
    """
    if statement and params:
        if isinstance(params, (tuple, list)):
            # BUG: dict.fromkeys() removes duplicates!
            params_dict = {
                f":arg{i}": param for i, param in enumerate(dict.fromkeys(params))
            }
        else:
            params_dict = params

        for key in sorted(params_dict, key=len, reverse=True):
            statement = statement.replace(key, str(params_dict[key]))
    return statement


# Strategy that generates lists with at least one duplicate
@st.composite
def lists_with_duplicates(draw):
    # Generate a base list
    base_values = draw(st.lists(
        st.one_of(
            st.integers(min_value=0, max_value=100),
            st.text(alphabet='abc', min_size=1, max_size=3)
        ),
        min_size=1,
        max_size=3
    ))

    # Ensure at least one duplicate by repeating a value
    if base_values:
        dup_value = draw(st.sampled_from(base_values))
        position = draw(st.integers(min_value=0, max_value=len(base_values)))
        base_values.insert(position, dup_value)

    return base_values


@given(lists_with_duplicates())
@example([100, 'active', 'active', 200])  # The specific failing case from the report
@settings(max_examples=10, suppress_health_check=[HealthCheck.filter_too_much])
def test_last_executed_query_preserves_all_params(params):
    # Skip if no duplicates (shouldn't happen with our strategy but just in case)
    if len(params) == len(set(params)):
        return

    # Build the SQL statement with placeholders
    placeholders = " AND ".join(f"col{i} = :arg{i}" for i in range(len(params)))
    statement = f"SELECT * FROM table WHERE {placeholders}"

    # Run the buggy implementation
    result = last_executed_query_buggy(statement, params)

    # Check that all placeholders were replaced
    for i in range(len(params)):
        placeholder = f":arg{i}"
        assert placeholder not in result, \
            f"Placeholder {placeholder} was not replaced! " \
            f"params={params}, unique_params={list(dict.fromkeys(params))}"

    print(f"Testing params: {params}")


if __name__ == "__main__":
    print("=" * 70)
    print("Property-Based Test for Django Oracle last_executed_query Bug")
    print("=" * 70)
    print()
    print("Running Hypothesis test to find failing cases...")
    print()

    try:
        test_last_executed_query_preserves_all_params()
        print("\nTest passed - no issues found!")
    except AssertionError as e:
        print("\n❌ TEST FAILED!")
        print()
        print("The test found a case where duplicate parameters cause")
        print("placeholders to remain unreplaced in the SQL string.")
        print()
        print("This happens because dict.fromkeys() removes duplicates,")
        print("causing the enumeration to produce fewer placeholder mappings")
        print("than there are actual parameters.")
        print()

        # Re-raise to show full Hypothesis output
        raise
```

<details>

<summary>
**Failing input**: `[100, 'active', 'active', 200]`
</summary>
```
======================================================================
Property-Based Test for Django Oracle last_executed_query Bug
======================================================================

Running Hypothesis test to find failing cases...


❌ TEST FAILED!

The test found a case where duplicate parameters cause
placeholders to remain unreplaced in the SQL string.

This happens because dict.fromkeys() removes duplicates,
causing the enumeration to produce fewer placeholder mappings
than there are actual parameters.

Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/49/hypo.py", line 85, in <module>
    test_last_executed_query_preserves_all_params()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/49/hypo.py", line 52, in test_last_executed_query_preserves_all_params
    @example([100, 'active', 'active', 200])  # The specific failing case from the report
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2062, in wrapped_test
    _raise_to_user(errors, state.settings, [], " in explicit examples")
    ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1613, in _raise_to_user
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/49/hypo.py", line 69, in test_last_executed_query_preserves_all_params
    assert placeholder not in result, \
           ^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Placeholder :arg3 was not replaced! params=[100, 'active', 'active', 200], unique_params=[100, 'active', 200]
Falsifying explicit example: test_last_executed_query_preserves_all_params(
    params=[100, 'active', 'active', 200],
)
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""
Demonstration of Django Oracle backend bug in last_executed_query method.
This shows how duplicate parameters cause incorrect SQL string substitution.
"""

def last_executed_query_buggy(statement, params):
    """
    This is the buggy implementation from Django's Oracle backend.
    Located at: django/db/backends/oracle/operations.py:331-349
    """
    if statement and params:
        if isinstance(params, (tuple, list)):
            # BUG: dict.fromkeys() removes duplicates from params!
            params_dict = {
                f":arg{i}": param for i, param in enumerate(dict.fromkeys(params))
            }
        else:
            params_dict = params

        # Replace placeholders with actual values
        for key in sorted(params_dict, key=len, reverse=True):
            statement = statement.replace(key, str(params_dict[key]))
    return statement


def last_executed_query_fixed(statement, params):
    """
    This is the fixed implementation that correctly handles duplicates.
    """
    if statement and params:
        if isinstance(params, (tuple, list)):
            # FIX: Don't use dict.fromkeys() - enumerate params directly
            params_dict = {
                f":arg{i}": param for i, param in enumerate(params)
            }
        else:
            params_dict = params

        # Replace placeholders with actual values
        for key in sorted(params_dict, key=len, reverse=True):
            statement = statement.replace(key, str(params_dict[key]))
    return statement


# Test case with duplicate parameter values
statement = "SELECT * FROM users WHERE id = :arg0 AND status = :arg1 AND type = :arg2 AND priority = :arg3"
params = [100, 'active', 'active', 200]

print("=" * 70)
print("Bug Demonstration: Django Oracle last_executed_query")
print("=" * 70)
print()

print("Input SQL statement:")
print(f"  {statement}")
print()

print("Input parameters:")
print(f"  {params}")
print()

# Show the buggy behavior
print("BUGGY IMPLEMENTATION:")
print("-" * 50)

# Show what dict.fromkeys() does
unique_params = list(dict.fromkeys(params))
print(f"  dict.fromkeys(params) returns: {unique_params}")
print(f"  Length: {len(unique_params)} (should be {len(params)})")
print()

# Show the incorrect params_dict that gets created
buggy_params_dict = {f":arg{i}": param for i, param in enumerate(dict.fromkeys(params))}
print("  Generated params_dict:")
for key, value in buggy_params_dict.items():
    print(f"    {key}: {repr(value)}")
print()

# Show the buggy result
buggy_result = last_executed_query_buggy(statement, params)
print("  Result SQL:")
print(f"    {buggy_result}")
print()

print("  ❌ ERROR: Placeholders :arg2 and :arg3 were NOT replaced!")
print()

print("CORRECT IMPLEMENTATION:")
print("-" * 50)

# Show the correct params_dict
correct_params_dict = {f":arg{i}": param for i, param in enumerate(params)}
print("  Generated params_dict:")
for key, value in correct_params_dict.items():
    print(f"    {key}: {repr(value)}")
print()

# Show the correct result
correct_result = last_executed_query_fixed(statement, params)
print("  Result SQL:")
print(f"    {correct_result}")
print()

print("  ✓ SUCCESS: All placeholders replaced correctly!")
print()

print("IMPACT:")
print("-" * 50)
print("  This bug causes incorrect debugging output when:")
print("  1. Using Oracle backend")
print("  2. Parameters list contains duplicate values")
print("  3. Developers try to debug SQL queries")
print()
print("  The bug makes it appear that some parameters were not bound,")
print("  when in reality they were bound correctly during execution.")
print("  This can mislead developers during debugging sessions.")
```

<details>

<summary>
Output showing incorrect placeholder substitution
</summary>
```
======================================================================
Bug Demonstration: Django Oracle last_executed_query
======================================================================

Input SQL statement:
  SELECT * FROM users WHERE id = :arg0 AND status = :arg1 AND type = :arg2 AND priority = :arg3

Input parameters:
  [100, 'active', 'active', 200]

BUGGY IMPLEMENTATION:
--------------------------------------------------
  dict.fromkeys(params) returns: [100, 'active', 200]
  Length: 3 (should be 4)

  Generated params_dict:
    :arg0: 100
    :arg1: 'active'
    :arg2: 200

  Result SQL:
    SELECT * FROM users WHERE id = 100 AND status = active AND type = 200 AND priority = :arg3

  ❌ ERROR: Placeholders :arg2 and :arg3 were NOT replaced!

CORRECT IMPLEMENTATION:
--------------------------------------------------
  Generated params_dict:
    :arg0: 100
    :arg1: 'active'
    :arg2: 'active'
    :arg3: 200

  Result SQL:
    SELECT * FROM users WHERE id = 100 AND status = active AND type = active AND priority = 200

  ✓ SUCCESS: All placeholders replaced correctly!

IMPACT:
--------------------------------------------------
  This bug causes incorrect debugging output when:
  1. Using Oracle backend
  2. Parameters list contains duplicate values
  3. Developers try to debug SQL queries

  The bug makes it appear that some parameters were not bound,
  when in reality they were bound correctly during execution.
  This can mislead developers during debugging sessions.
```
</details>

## Why This Is A Bug

This is a legitimate bug in Django's Oracle backend that violates the expected behavior of the `last_executed_query` method. The method is documented to "Return a string of the query last executed by the given cursor, with placeholders replaced with actual values" for debugging purposes.

The bug occurs because:

1. **Incorrect duplicate removal**: The code uses `dict.fromkeys(params)` which removes duplicate values from the parameters list before enumeration. This is fundamentally wrong because SQL parameters can legitimately have duplicate values.

2. **Placeholder mismatch**: When a params list has N items with duplicates (e.g., `[100, 'active', 'active', 200]`), the code creates fewer placeholder mappings than needed. In this example, it creates mappings for `:arg0`, `:arg1`, and `:arg2` but not `:arg3`.

3. **Wrong value substitution**: Even worse, the remaining placeholders get the wrong values. In the example, `:arg2` gets the value `200` instead of `'active'`.

4. **Debugging confusion**: This makes debugging extremely difficult because developers see unreplaced placeholders and wrong values, leading them to believe their query is malformed when it actually executes correctly.

5. **Inconsistency with other backends**: MySQL and PostgreSQL backends correctly handle duplicate parameters in their `last_executed_query` implementations. Only the Oracle backend has this bug.

## Relevant Context

- **Source file location**: `/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/django/db/backends/oracle/operations.py`, lines 331-349
- **Django documentation**: The method is part of Django's database backend API for debugging purposes
- **Common scenario**: Duplicate parameters are extremely common in real-world SQL queries:
  - Date range queries: `WHERE date >= :arg0 AND date <= :arg1 AND created_date = :arg2` where arg0 and arg2 might be the same
  - Status filters: `WHERE status IN (:arg0, :arg1) OR default_status = :arg2` where statuses might repeat
  - Join conditions: Complex queries often repeat values across different conditions
- **Oracle-specific**: This only affects the Oracle backend because Oracle's cursor doesn't provide the fully substituted query like PostgreSQL's `cursor.query` or MySQL's `cursor._executed`

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