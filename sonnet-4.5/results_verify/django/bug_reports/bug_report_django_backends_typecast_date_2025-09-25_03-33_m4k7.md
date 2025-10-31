# Bug Report: django.db.backends.utils.typecast_date Poor Error Handling

**Target**: `django.db.backends.utils.typecast_date`, `django.db.backends.utils.typecast_timestamp`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `typecast_date` and `typecast_timestamp` functions crash with confusing `TypeError` messages when given strings that don't match the expected format, instead of raising a clear `ValueError` or returning `None`.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings
from django.db.backends.utils import typecast_date, typecast_timestamp


@given(st.text())
@settings(max_examples=500)
def test_typecast_timestamp_doesnt_crash(s):
    try:
        result = typecast_timestamp(s)
        assert result is None or isinstance(result, (datetime.date, datetime.datetime))
    except ValueError:
        pass
```

**Failing input**: `s='0'`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from django.db.backends.utils import typecast_date, typecast_timestamp

typecast_date('0')

typecast_date('2024')

typecast_timestamp('0')
```

## Why This Is A Bug

The functions handle empty/None strings gracefully by returning `None`, but crash with a confusing `TypeError: function missing required argument 'month' (pos 2)` for malformed strings. The error message exposes implementation details (that it's unpacking into `datetime.date`) rather than clearly indicating the input format is invalid.

Expected behavior: Either raise `ValueError` with a clear message about the expected format, or return `None` for any malformed input (consistent with how empty strings are handled).

## Fix

```diff
diff --git a/django/db/backends/utils.py b/django/db/backends/utils.py
index 1234567..abcdefg 100644
--- a/django/db/backends/utils.py
+++ b/django/db/backends/utils.py
@@ -213,9 +213,13 @@ def typecast_date(s):

 def typecast_date(s):
-    return (
-        datetime.date(*map(int, s.split("-"))) if s else None
-    )  # return None if s is null
+    if not s:
+        return None
+    parts = s.split("-")
+    if len(parts) != 3:
+        return None
+    try:
+        return datetime.date(*map(int, parts))
+    except (ValueError, OverflowError):
+        return None
```