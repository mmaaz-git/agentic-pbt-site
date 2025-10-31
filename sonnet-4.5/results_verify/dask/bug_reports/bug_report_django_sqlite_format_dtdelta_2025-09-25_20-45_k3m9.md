# Bug Report: Django SQLite format_dtdelta Inconsistent Return Types

**Target**: `django.db.backends.sqlite3._functions._sqlite_format_dtdelta`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_sqlite_format_dtdelta` function returns inconsistent types: `str` for addition/subtraction operations, but `float` for multiplication/division operations. This violates the function's intended purpose as indicated by its name ("format") and creates type inconsistency in SQL query results.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings
from django.db.backends.sqlite3._functions import _sqlite_format_dtdelta


@given(st.floats(min_value=0.1, max_value=1e6), st.floats(min_value=0.1, max_value=1e6))
@settings(max_examples=100)
def test_format_dtdelta_always_returns_string(lhs, rhs):
    for connector in ["+", "-", "*", "/"]:
        result = _sqlite_format_dtdelta(connector, lhs, rhs)
        if result is not None:
            assert isinstance(result, str), f"format_dtdelta({connector!r}, {lhs}, {rhs}) should return string, got {type(result)}"
```

**Failing input**: `_sqlite_format_dtdelta("*", 2.5, 3.0)` returns `7.5` (float) instead of `"7.5"` (string)

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from django.db.backends.sqlite3._functions import _sqlite_format_dtdelta

print("Addition returns:", type(_sqlite_format_dtdelta("+", 1000000, 2000000)))
print("Subtraction returns:", type(_sqlite_format_dtdelta("-", 2000000, 1000000)))
print("Multiplication returns:", type(_sqlite_format_dtdelta("*", 2.5, 3.0)))
print("Division returns:", type(_sqlite_format_dtdelta("/", 6.0, 2.0)))
```

Expected output:
```
Addition returns: <class 'str'>
Subtraction returns: <class 'str'>
Multiplication returns: <class 'str'>
Division returns: <class 'str'>
```

Actual output:
```
Addition returns: <class 'str'>
Subtraction returns: <class 'str'>
Multiplication returns: <class 'float'>
Division returns: <class 'float'>
```

## Why This Is A Bug

1. **Function name implies formatting**: The function is named `_sqlite_format_dtdelta` (emphasis on "format"), suggesting it should return formatted strings.

2. **Code comment indicates string output**: The comment in the function states:
   ```python
   # It will be formatted as "%Y-%m-%d" or "%Y-%m-%d %H:%M:%S[.%f]"
   ```
   This clearly indicates formatted string output is expected.

3. **Inconsistent behavior**: Two of four code paths call `str()`, while two don't. This inconsistency suggests the missing `str()` calls are bugs, not intentional design.

4. **SQLite function consistency**: Functions registered with SQLite's `create_function` should have consistent return types to avoid type confusion in SQL queries.

## Fix

```diff
--- a/django/db/backends/sqlite3/_functions.py
+++ b/django/db/backends/sqlite3/_functions.py
@@ -264,9 +264,9 @@ def _sqlite_format_dtdelta(connector, lhs, rhs):
     if connector == "+":
         out = str(real_lhs + real_rhs)
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