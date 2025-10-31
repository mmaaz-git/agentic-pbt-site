# Bug Report: django.core.mail.backends.base Context Manager Masks Exceptions

**Target**: `django.core.mail.backends.base.BaseEmailBackend.__enter__`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `BaseEmailBackend.__enter__()` method masks the original exception from `open()` when `close()` also raises an exception, making debugging difficult and violating Python exception handling best practices.

## Property-Based Test

```python
from django.core.mail.backends.base import BaseEmailBackend

def test_base_context_manager_exception_handling():
    class FailingBackend(BaseEmailBackend):
        def open(self):
            raise ValueError("Open failed")

        def close(self):
            raise RuntimeError("Close failed")

        def send_messages(self, email_messages):
            return 0

    backend = FailingBackend()
    try:
        with backend:
            pass
        assert False, "Should have raised an exception"
    except ValueError:
        pass
    except RuntimeError:
        raise AssertionError("RuntimeError from close() masked the ValueError from open()")
```

**Failing input**: Any backend where both `open()` and `close()` raise exceptions.

## Reproducing the Bug

```python
from django.core.mail.backends.base import BaseEmailBackend

class FailingBackend(BaseEmailBackend):
    def open(self):
        raise ValueError("Open failed")

    def close(self):
        raise RuntimeError("Close failed")

    def send_messages(self, email_messages):
        return 0

backend = FailingBackend()
try:
    with backend:
        pass
except Exception as e:
    print(f"Caught: {type(e).__name__}: {e}")
```

Output:
```
Caught: RuntimeError: Close failed
```

Expected: Should raise `ValueError: Open failed` (the original error), not the secondary error from cleanup.

## Why This Is A Bug

When `open()` fails in `__enter__()`, the code calls `close()` for cleanup (line 48 in base.py). However, if `close()` also raises an exception, it replaces the original exception from `open()`. This:

1. Violates Python's exception handling best practices (PEP 343 and exception chaining)
2. Makes debugging harder by hiding the root cause
3. Is inconsistent with modern Python context manager patterns that use exception chaining

## Fix

```diff
--- a/django/core/mail/backends/base.py
+++ b/django/core/mail/backends/base.py
@@ -44,8 +44,11 @@ class BaseEmailBackend:
     def __enter__(self):
         try:
             self.open()
-        except Exception:
-            self.close()
+        except Exception as exc:
+            try:
+                self.close()
+            except Exception:
+                pass
             raise
         return self
```