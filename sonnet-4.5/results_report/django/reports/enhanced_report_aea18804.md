# Bug Report: django.core.mail.backends.filebased Confusing TypeError on Missing File Path Configuration

**Target**: `django.core.mail.backends.filebased.EmailBackend.__init__`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The file-based email backend raises a confusing `TypeError` instead of the expected `ImproperlyConfigured` error when neither the `file_path` parameter nor the `EMAIL_FILE_PATH` setting is provided, violating the established error handling pattern within the same method.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings
from django.core.exceptions import ImproperlyConfigured
from django.core.mail.backends.filebased import EmailBackend
from unittest.mock import patch, MagicMock


@given(st.none())
@settings(max_examples=10)
def test_filebased_backend_validates_path(none_value):
    mock_settings = MagicMock()
    del mock_settings.EMAIL_FILE_PATH

    with patch('django.core.mail.backends.filebased.settings', mock_settings):
        try:
            backend = EmailBackend(file_path=none_value)
            assert False, "Should have raised an exception"
        except ImproperlyConfigured:
            pass
        except TypeError:
            assert False, "Should raise ImproperlyConfigured, not TypeError"

if __name__ == "__main__":
    test_filebased_backend_validates_path()
```

<details>

<summary>
**Failing input**: `none_value=None`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/11/hypo.py", line 18, in test_filebased_backend_validates_path
    backend = EmailBackend(file_path=none_value)
  File "/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/django/core/mail/backends/filebased.py", line 18, in __init__
    self.file_path = os.path.abspath(self.file_path)
                     ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^
  File "<frozen posixpath>", line 378, in abspath
TypeError: expected str, bytes or os.PathLike object, not NoneType

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/11/hypo.py", line 26, in <module>
    test_filebased_backend_validates_path()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/11/hypo.py", line 11, in test_filebased_backend_validates_path
    @settings(max_examples=10)
                   ^^^
  File "/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/11/hypo.py", line 23, in test_filebased_backend_validates_path
    assert False, "Should raise ImproperlyConfigured, not TypeError"
           ^^^^^
AssertionError: Should raise ImproperlyConfigured, not TypeError
Falsifying example: test_filebased_backend_validates_path(
    none_value=None,
)
```
</details>

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

import os
from unittest.mock import patch, MagicMock
from django.core.mail.backends.filebased import EmailBackend
from django.core.exceptions import ImproperlyConfigured

# Case 1: file_path=None with no EMAIL_FILE_PATH setting
print("Case 1: file_path=None with no EMAIL_FILE_PATH setting")
print("-" * 50)
mock_settings = MagicMock()
del mock_settings.EMAIL_FILE_PATH

with patch('django.core.mail.backends.filebased.settings', mock_settings):
    try:
        backend = EmailBackend(file_path=None)
        print("No error raised - this shouldn't happen!")
    except ImproperlyConfigured as e:
        print(f"ImproperlyConfigured: {e}")
    except TypeError as e:
        print(f"TypeError: {e}")
        print(f"Error type: {type(e).__name__}")

print("\n")

# Case 2: No arguments with no EMAIL_FILE_PATH setting
print("Case 2: No arguments with no EMAIL_FILE_PATH setting")
print("-" * 50)
with patch('django.core.mail.backends.filebased.settings', mock_settings):
    try:
        backend = EmailBackend()
        print("No error raised - this shouldn't happen!")
    except ImproperlyConfigured as e:
        print(f"ImproperlyConfigured: {e}")
    except TypeError as e:
        print(f"TypeError: {e}")
        print(f"Error type: {type(e).__name__}")
```

<details>

<summary>
TypeError raised instead of ImproperlyConfigured
</summary>
```
Case 1: file_path=None with no EMAIL_FILE_PATH setting
--------------------------------------------------
TypeError: expected str, bytes or os.PathLike object, not NoneType
Error type: TypeError


Case 2: No arguments with no EMAIL_FILE_PATH setting
--------------------------------------------------
TypeError: expected str, bytes or os.PathLike object, not NoneType
Error type: TypeError
```
</details>

## Why This Is A Bug

This violates Django's established error handling contract within the same method. The `EmailBackend.__init__` method consistently uses `ImproperlyConfigured` exceptions for all other configuration-related errors:

1. **Lines 21-25**: Raises `ImproperlyConfigured` when the path exists but is not a directory
2. **Lines 26-30**: Raises `ImproperlyConfigured` when directory creation fails
3. **Lines 32-35**: Raises `ImproperlyConfigured` when the directory is not writable

However, when no file path is provided at all (either through the `file_path` parameter or `EMAIL_FILE_PATH` setting), the code fails at line 18 with a generic `TypeError` from `os.path.abspath(None)`. This creates an inconsistent error handling pattern where:
- Configuration errors that occur after path validation raise `ImproperlyConfigured` with helpful messages
- The most basic configuration error (no path provided) raises an unhelpful `TypeError`

Users receive the cryptic error "expected str, bytes or os.PathLike object, not NoneType" instead of a clear message like "EMAIL_FILE_PATH setting must be configured or file_path parameter must be provided."

## Relevant Context

The Django documentation for the file-based email backend mentions that the directory can be specified via:
- The `EMAIL_FILE_PATH` setting in Django configuration
- The `file_path` keyword argument when creating a connection

However, the documentation doesn't explicitly state what happens when neither is provided. The code's error handling pattern strongly suggests all configuration errors should raise `ImproperlyConfigured`, making this TypeError an oversight in the implementation.

This issue likely affects developers during initial setup or testing when they haven't yet configured the email backend properly. While the workaround is simple (always provide a file path), the inconsistent error handling makes debugging unnecessarily confusing.

## Proposed Fix

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