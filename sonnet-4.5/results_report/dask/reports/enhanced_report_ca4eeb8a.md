# Bug Report: Django SQLite _sqlite_format_dtdelta Inconsistent Return Types

**Target**: `django.db.backends.sqlite3._functions._sqlite_format_dtdelta`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_sqlite_format_dtdelta` function returns inconsistent types across different operations: it returns `str` for addition and subtraction, but `float` for multiplication and division, violating type consistency expectations for a formatting function.

## Property-Based Test

```python
#!/usr/bin/env python3
"""
Property-based test using Hypothesis to verify that _sqlite_format_dtdelta
always returns strings, regardless of the operation type.
"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings
from django.db.backends.sqlite3._functions import _sqlite_format_dtdelta


@given(st.floats(min_value=0.1, max_value=1e6), st.floats(min_value=0.1, max_value=1e6))
@settings(max_examples=100)
def test_format_dtdelta_always_returns_string(lhs, rhs):
    """Test that _sqlite_format_dtdelta always returns a string for all operations."""
    for connector in ["+", "-", "*", "/"]:
        result = _sqlite_format_dtdelta(connector, lhs, rhs)
        if result is not None:
            assert isinstance(result, str), f"format_dtdelta({connector!r}, {lhs}, {rhs}) should return string, got {type(result)}"


if __name__ == "__main__":
    # Run the property-based test
    try:
        test_format_dtdelta_always_returns_string()
        print("All tests passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")
        print("\nThis demonstrates the bug where multiplication (*) and division (/)")
        print("operations return float values instead of formatted strings.")
```

<details>

<summary>
**Failing input**: `_sqlite_format_dtdelta('*', 1.0, 1.0)`
</summary>
```
Test failed: format_dtdelta('*', 1.0, 1.0) should return string, got <class 'float'>

This demonstrates the bug where multiplication (*) and division (/)
operations return float values instead of formatted strings.
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""
Minimal reproduction of the Django SQLite format_dtdelta bug.
Demonstrates inconsistent return types: str for +/- but float for *// operations.
"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from django.db.backends.sqlite3._functions import _sqlite_format_dtdelta

print("Testing _sqlite_format_dtdelta return types:\n")

# Test addition
result_add = _sqlite_format_dtdelta("+", 1000000, 2000000)
print(f"Addition (+):")
print(f"  Input: _sqlite_format_dtdelta('+', 1000000, 2000000)")
print(f"  Result: {result_add!r}")
print(f"  Type: {type(result_add)}")
print()

# Test subtraction
result_sub = _sqlite_format_dtdelta("-", 2000000, 1000000)
print(f"Subtraction (-):")
print(f"  Input: _sqlite_format_dtdelta('-', 2000000, 1000000)")
print(f"  Result: {result_sub!r}")
print(f"  Type: {type(result_sub)}")
print()

# Test multiplication
result_mul = _sqlite_format_dtdelta("*", 2.5, 3.0)
print(f"Multiplication (*):")
print(f"  Input: _sqlite_format_dtdelta('*', 2.5, 3.0)")
print(f"  Result: {result_mul!r}")
print(f"  Type: {type(result_mul)}")
print()

# Test division
result_div = _sqlite_format_dtdelta("/", 6.0, 2.0)
print(f"Division (/):")
print(f"  Input: _sqlite_format_dtdelta('/', 6.0, 2.0)")
print(f"  Result: {result_div!r}")
print(f"  Type: {type(result_div)}")
print()

# Summary
print("=" * 50)
print("SUMMARY:")
print("=" * 50)
print(f"Addition returns:       {type(result_add).__name__}")
print(f"Subtraction returns:    {type(result_sub).__name__}")
print(f"Multiplication returns: {type(result_mul).__name__}")
print(f"Division returns:       {type(result_div).__name__}")
print()
print("Expected: All operations should return 'str' type")
print("Actual:   Addition and subtraction return 'str',")
print("          but multiplication and division return 'float'")
```

<details>

<summary>
Inconsistent return types: string for +/- operations, float for *// operations
</summary>
```
Testing _sqlite_format_dtdelta return types:

Addition (+):
  Input: _sqlite_format_dtdelta('+', 1000000, 2000000)
  Result: '0:00:03'
  Type: <class 'str'>

Subtraction (-):
  Input: _sqlite_format_dtdelta('-', 2000000, 1000000)
  Result: '0:00:01'
  Type: <class 'str'>

Multiplication (*):
  Input: _sqlite_format_dtdelta('*', 2.5, 3.0)
  Result: 7.5
  Type: <class 'float'>

Division (/):
  Input: _sqlite_format_dtdelta('/', 6.0, 2.0)
  Result: 3.0
  Type: <class 'float'>

==================================================
SUMMARY:
==================================================
Addition returns:       str
Subtraction returns:    str
Multiplication returns: float
Division returns:       float

Expected: All operations should return 'str' type
Actual:   Addition and subtraction return 'str',
          but multiplication and division return 'float'
```
</details>

## Why This Is A Bug

This violates expected behavior in multiple ways:

1. **Function naming indicates formatting intent**: The function is explicitly named `_sqlite_format_dtdelta` where "format" strongly implies string output. Functions with "format" in their name conventionally return formatted strings, not raw numeric values.

2. **Code comment contradicts actual behavior**: Lines 265-266 in the source code contain the comment: "It will be formatted as '%Y-%m-%d' or '%Y-%m-%d %H:%M:%S[.%f]'". This comment appears immediately before the addition operation and clearly indicates that formatting to a string is the intended behavior for all operations.

3. **Partial implementation reveals oversight**: The code explicitly calls `str()` for addition (line 267) and subtraction (line 269), but omits this conversion for multiplication (line 271) and division (line 273). This inconsistency strongly suggests the `str()` calls were forgotten rather than intentionally omitted.

4. **Type inconsistency breaks composability**: When registered as a SQLite user-defined function, inconsistent return types can cause unexpected behavior in SQL queries. Code that expects string results from this formatting function will fail when multiplication or division is used.

5. **Violates principle of least surprise**: Users of a formatting function reasonably expect consistent return types across all operations, especially when the function's primary purpose is to format datetime deltas for SQLite operations.

## Relevant Context

The function is located at `/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/django/db/backends/sqlite3/_functions.py` lines 249-274.

This function is registered with SQLite as a user-defined function named `django_format_dtdelta` with 3 parameters. It's used internally by Django's SQLite backend for datetime delta calculations in ORM operations.

The function handles four arithmetic operations (+, -, *, /) on datetime-related values. For addition and subtraction, it processes the parameters through `_sqlite_prepare_dtdelta_param` which converts them to timedelta objects when appropriate. For all operations, the function is supposed to return formatted string representations.

Key code section (lines 264-274):
- Lines 264-266: Comment indicates formatting will occur
- Line 267: Addition explicitly converts to string with `str(real_lhs + real_rhs)`
- Line 269: Subtraction explicitly converts to string with `str(real_lhs - real_rhs)`
- Line 271: Multiplication returns raw float: `real_lhs * real_rhs` (missing `str()`)
- Line 273: Division returns raw float: `real_lhs / real_rhs` (missing `str()`)

This bug affects Django applications using SQLite as their database backend when performing datetime delta calculations involving multiplication or division operations.

## Proposed Fix

```diff
--- a/django/db/backends/sqlite3/_functions.py
+++ b/django/db/backends/sqlite3/_functions.py
@@ -268,9 +268,9 @@ def _sqlite_format_dtdelta(connector, lhs, rhs):
     elif connector == "-":
         out = str(real_lhs - real_rhs)
     elif connector == "*":
-        out = real_lhs * real_rhs
+        out = str(real_lhs * real_rhs)
     else:
-        out = real_lhs / real_rhs
+        out = str(real_lhs / real_rhs)
     return out
```