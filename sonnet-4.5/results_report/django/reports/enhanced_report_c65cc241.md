# Bug Report: Django Filebased Email Backend Crashes with None Path

**Target**: `django.core.mail.backends.filebased.EmailBackend`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The Django filebased email backend crashes with an unhelpful TypeError when both the `file_path` parameter and `settings.EMAIL_FILE_PATH` are None, instead of raising a clear ImproperlyConfigured exception as is the Django pattern for missing configuration.

## Property-Based Test

```python
#!/usr/bin/env python3
"""Property-based test for Django filebased email backend None path handling"""

from hypothesis import given, strategies as st, settings as hypo_settings
from django.core.mail.backends.filebased import EmailBackend
from django.conf import settings
import traceback

# Configure Django with EMAIL_FILE_PATH set to None
if not settings.configured:
    settings.configure(EMAIL_FILE_PATH=None, DEFAULT_CHARSET='utf-8')

@given(st.just(None))
@hypo_settings(max_examples=1, deadline=None)
def test_filebased_none_path_handling(file_path):
    """Test that filebased backend handles None file_path gracefully"""
    print(f"Testing with file_path={file_path}")
    try:
        backend = EmailBackend(file_path=file_path)
        print("ERROR: Backend created successfully - expected TypeError!")
        assert False, "Should raise exception for None file_path"
    except TypeError as e:
        print(f"Caught expected TypeError: {e}")
        traceback.print_exc()
        # Expected behavior with current bug

if __name__ == "__main__":
    # Run the property test
    print("Running property-based test for filebased email backend with None path...")
    test_filebased_none_path_handling()
    print("\nTest completed - TypeError was caught as expected (demonstrating the bug)")
```

<details>

<summary>
**Failing input**: `file_path=None`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/34/hypo.py", line 19, in test_filebased_none_path_handling
    backend = EmailBackend(file_path=file_path)
  File "/home/npc/miniconda/lib/python3.13/site-packages/django/core/mail/backends/filebased.py", line 18, in __init__
    self.file_path = os.path.abspath(self.file_path)
                     ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^
  File "<frozen posixpath>", line 378, in abspath
TypeError: expected str, bytes or os.PathLike object, not NoneType
Running property-based test for filebased email backend with None path...
Testing with file_path=None
Caught expected TypeError: expected str, bytes or os.PathLike object, not NoneType

Test completed - TypeError was caught as expected (demonstrating the bug)
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""Minimal reproduction of Django filebased email backend None path crash"""

from django.conf import settings
from django.core.mail.backends.filebased import EmailBackend

# Configure Django with EMAIL_FILE_PATH set to None
if not settings.configured:
    settings.configure(EMAIL_FILE_PATH=None, DEFAULT_CHARSET='utf-8')

# This should crash with TypeError when both file_path and EMAIL_FILE_PATH are None
try:
    backend = EmailBackend(file_path=None)
    print("ERROR: Expected TypeError but backend was created successfully")
except TypeError as e:
    print(f"Caught expected TypeError: {e}")
    import traceback
    traceback.print_exc()
```

<details>

<summary>
TypeError crash when instantiating EmailBackend with None path
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/34/repo.py", line 13, in <module>
    backend = EmailBackend(file_path=None)
  File "/home/npc/miniconda/lib/python3.13/site-packages/django/core/mail/backends/filebased.py", line 18, in __init__
    self.file_path = os.path.abspath(self.file_path)
                     ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^
  File "<frozen posixpath>", line 378, in abspath
TypeError: expected str, bytes or os.PathLike object, not NoneType
Caught expected TypeError: expected str, bytes or os.PathLike object, not NoneType
```
</details>

## Why This Is A Bug

This violates Django's established patterns for configuration error handling. The filebased email backend already imports and uses `ImproperlyConfigured` for other validation errors (lines 22-35 in filebased.py), but fails to validate that the file path is not None before calling `os.path.abspath()` on line 18.

Looking at the code structure:
- Line 12: The `file_path` parameter defaults to `None`
- Line 17: Uses `getattr(settings, "EMAIL_FILE_PATH", None)` which explicitly provides `None` as a default
- Line 18: Immediately calls `os.path.abspath(self.file_path)` without checking if it's None

This results in a generic `TypeError` that doesn't help developers understand what configuration is missing. Django's pattern throughout the framework is to raise `ImproperlyConfigured` with a clear message explaining what setting needs to be configured.

## Relevant Context

The filebased email backend is commonly used in development environments to write emails to files instead of sending them. The backend already has proper error handling for other configuration issues:
- When the path exists but is not a directory (lines 21-25)
- When the directory cannot be created (lines 26-30)
- When the directory is not writable (lines 32-35)

All of these cases properly raise `ImproperlyConfigured` with helpful error messages. The missing None check is inconsistent with this existing validation pattern.

Django documentation: The `EMAIL_FILE_PATH` setting is not documented as a required setting, and the code structure suggests None should be handled gracefully.

Code location: `/django/core/mail/backends/filebased.py:18`

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
+                "File-based email backend requires file_path parameter or EMAIL_FILE_PATH setting"
+            )
         self.file_path = os.path.abspath(self.file_path)
         try:
             os.makedirs(self.file_path, exist_ok=True)
```