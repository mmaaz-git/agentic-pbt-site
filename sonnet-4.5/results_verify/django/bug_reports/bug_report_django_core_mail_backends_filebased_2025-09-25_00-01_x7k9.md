# Bug Report: django.core.mail.backends.filebased Unclear Error on Missing Path

**Target**: `django.core.mail.backends.filebased.EmailBackend.__init__`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The file-based email backend raises a confusing `TypeError` instead of a clear `ImproperlyConfigured` error when neither the `file_path` parameter nor the `EMAIL_FILE_PATH` setting is provided.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

from hypothesis import given, strategies as st, settings
from django.core.exceptions import ImproperlyConfigured
from django.core.mail.backends.filebased import EmailBackend
from unittest.mock import patch


@given(st.none())
@settings(max_examples=10)
def test_filebased_backend_validates_path(none_value):
    with patch('django.conf.settings') as mock_settings:
        del mock_settings.EMAIL_FILE_PATH

        try:
            backend = EmailBackend(file_path=none_value)
            assert False, "Should have raised an exception"
        except ImproperlyConfigured:
            pass
        except TypeError:
            assert False, "Should raise ImproperlyConfigured, not TypeError"
```

**Failing input**: `file_path=None` with no `EMAIL_FILE_PATH` setting

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

import os
from unittest.mock import patch, MagicMock
from django.core.mail.backends.filebased import EmailBackend

mock_settings = MagicMock()
del mock_settings.EMAIL_FILE_PATH

with patch('django.core.mail.backends.filebased.settings', mock_settings):
    try:
        backend = EmailBackend(file_path=None)
    except TypeError as e:
        print(f"TypeError: {e}")
```

## Why This Is A Bug

1. **Inconsistent Error Handling**: The backend properly validates directory existence and writability with `ImproperlyConfigured` exceptions (lines 21-35), but fails to validate that a path was provided at all.

2. **Confusing Error Message**: Users get `TypeError: expected str, bytes or os.PathLike object, not NoneType` instead of a clear message like "EMAIL_FILE_PATH setting is required or file_path parameter must be provided".

3. **Violates Error Handling Pattern**: The code establishes a pattern of using `ImproperlyConfigured` for configuration errors, but this case falls through to a `TypeError`.

## Fix

```diff
--- a/django/core/mail/backends/filebased.py
+++ b/django/core/mail/backends/filebased.py
@@ -15,6 +15,10 @@ class EmailBackend(ConsoleEmailBackend):
             self.file_path = file_path
         else:
             self.file_path = getattr(settings, "EMAIL_FILE_PATH", None)
+        if self.file_path is None:
+            raise ImproperlyConfigured(
+                "EMAIL_FILE_PATH setting must be configured or file_path parameter must be provided."
+            )
         self.file_path = os.path.abspath(self.file_path)
         try:
             os.makedirs(self.file_path, exist_ok=True)
```