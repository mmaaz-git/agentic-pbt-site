# Bug Report: django.db.backends.sqlite3._functions._sqlite_sqrt Domain Error

**Target**: `django.db.backends.sqlite3._functions._sqlite_sqrt`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `_sqlite_sqrt` function crashes with an unhandled `ValueError` when given negative inputs, instead of gracefully returning `None` to represent SQL NULL as other database systems do for invalid mathematical operations.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from django.db.backends.sqlite3._functions import _sqlite_sqrt


@given(st.floats(allow_nan=False, allow_infinity=False, min_value=-100, max_value=0))
def test_sqrt_negative(x):
    result = _sqlite_sqrt(x)
```

**Failing input**: `x=-1.0` (or any negative value)

## Reproducing the Bug

```python
from django.db.backends.sqlite3._functions import _sqlite_sqrt

result = _sqlite_sqrt(-1)
```

## Why This Is A Bug

This function is registered as a SQL function (SQRT) for SQLite and should handle domain errors gracefully. Taking the square root of a negative number in real arithmetic is undefined, and SQL databases handle this by returning NULL rather than crashing. The current implementation crashes with an unhandled Python `ValueError: math domain error`, which would cause the SQLite backend to fail when executing a query like `SELECT SQRT(-1)`.

Other database systems handle sqrt of negative numbers gracefully:
- PostgreSQL: `SELECT SQRT(-1)` returns NaN
- MySQL: `SELECT SQRT(-1)` returns NULL
- SQLite 3.35+ (native): `SELECT SQRT(-1)` returns NULL

The same issue affects other math functions with restricted domains: `_sqlite_ln` (needs x > 0), `_sqlite_log` (needs x > 0 and base > 0, base ≠ 1), `_sqlite_asin` (needs |x| ≤ 1), and `_sqlite_acos` (needs |x| ≤ 1).

## Fix

```diff
--- a/django/db/backends/sqlite3/_functions.py
+++ b/django/db/backends/sqlite3/_functions.py
@@ -486,7 +486,9 @@ def _sqlite_sin(x):
 def _sqlite_sqrt(x):
     if x is None:
         return None
-    return sqrt(x)
+    if x < 0:
+        return None
+    return sqrt(x)
```