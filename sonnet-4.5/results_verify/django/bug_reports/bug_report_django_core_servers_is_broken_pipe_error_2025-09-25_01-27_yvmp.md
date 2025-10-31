# Bug Report: django.core.servers.basehttp.is_broken_pipe_error Crashes Without Active Exception

**Target**: `django.core.servers.basehttp.is_broken_pipe_error`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `is_broken_pipe_error()` function crashes with `TypeError` when called without an active exception, because it attempts to call `issubclass()` on `None`.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from django.core.servers import basehttp


def test_is_broken_pipe_error_no_active_exception():
    result = basehttp.is_broken_pipe_error()
    assert result == False, "Should return False when no exception is active"
```

**Failing input**: No active exception (i.e., `sys.exc_info()` returns `(None, None, None)`)

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

from django.core.servers.basehttp import is_broken_pipe_error

result = is_broken_pipe_error()
```

**Output**:
```
TypeError: issubclass() arg 1 must be a class
```

## Why This Is A Bug

The function `is_broken_pipe_error()` is designed to check if the current exception is a broken pipe or connection error. When there's no active exception, `sys.exc_info()` returns `(None, None, None)`, so `exc_type` is `None`. Calling `issubclass(None, ...)` raises a `TypeError`.

The function should defensively handle the case where there's no active exception and return `False`, since the question "is the current error a broken pipe error?" should naturally return `False` when there is no error.

While the function is currently only called within exception handlers (where an exception is guaranteed to be active), the lack of defensive programming makes the code fragile and prone to errors if:
1. The function is refactored or reused elsewhere
2. The calling context changes
3. A developer mistakenly calls it outside an exception handler

## Fix

```diff
--- a/django/core/servers/basehttp.py
+++ b/django/core/servers/basehttp.py
@@ -55,6 +55,8 @@ def get_internal_wsgi_application():

 def is_broken_pipe_error():
     exc_type, _, _ = sys.exc_info()
+    if exc_type is None:
+        return False
     return issubclass(
         exc_type,
         (
```