# Bug Report: Django Mail Filebased Backend None Path Crash

**Target**: `django.core.mail.backends.filebased.EmailBackend`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The filebased email backend crashes with TypeError when both the `file_path` parameter and `settings.EMAIL_FILE_PATH` are None, due to calling `os.path.abspath(None)`.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from django.core.mail.backends.filebased import EmailBackend
from django.conf import settings

@given(st.just(None))
def test_filebased_none_path_handling(file_path):
    if not settings.configured:
        settings.configure(EMAIL_FILE_PATH=None, DEFAULT_CHARSET='utf-8')

    try:
        backend = EmailBackend(file_path=file_path)
        assert False, "Should raise exception for None file_path"
    except TypeError:
        pass
```

**Failing input**: `file_path=None` when `settings.EMAIL_FILE_PATH` is also `None`

## Reproducing the Bug

```python
from django.conf import settings
from django.core.mail.backends.filebased import EmailBackend

if not settings.configured:
    settings.configure(EMAIL_FILE_PATH=None, DEFAULT_CHARSET='utf-8')

backend = EmailBackend(file_path=None)
```

Output:
```
TypeError: expected str, bytes or os.PathLike object, not NoneType
```

## Why This Is A Bug

Looking at `filebased.py` lines 14-18:

```python
if file_path is not None:
    self.file_path = file_path
else:
    self.file_path = getattr(settings, "EMAIL_FILE_PATH", None)
self.file_path = os.path.abspath(self.file_path)  # Line 18
```

When both `file_path` parameter and `settings.EMAIL_FILE_PATH` are `None`:
1. Line 17 sets `self.file_path = None`
2. Line 18 calls `os.path.abspath(None)`
3. This raises `TypeError` because `os.path.abspath()` expects a path-like object, not `None`

The code should validate that `file_path` is not `None` before calling `os.path.abspath()`, or provide a more helpful error message.

## Fix

```diff
--- a/django/core/mail/backends/filebased.py
+++ b/django/core/mail/backends/filebased.py
@@ -14,6 +14,10 @@ class EmailBackend(ConsoleEmailBackend):
         if file_path is not None:
             self.file_path = file_path
         else:
             self.file_path = getattr(settings, "EMAIL_FILE_PATH", None)
+        if self.file_path is None:
+            raise ImproperlyConfigured(
+                "file_path must be provided or EMAIL_FILE_PATH setting must be configured"
+            )
         self.file_path = os.path.abspath(self.file_path)
```