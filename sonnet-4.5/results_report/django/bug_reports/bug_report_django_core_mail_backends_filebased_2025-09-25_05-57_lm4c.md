# Bug Report: django.core.mail.backends.filebased Empty Messages Returns None

**Target**: `django.core.mail.backends.filebased.EmailBackend.send_messages`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The filebased email backend returns `None` instead of `0` when `send_messages()` is called with an empty list, violating the documented API contract that states it should "return the number of email messages sent." This bug is inherited from the console backend.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from django.core.mail.backends.filebased import EmailBackend as FileBasedBackend
import tempfile

@given(st.booleans())
def test_filebased_backend_empty_messages_returns_int(fail_silently):
    with tempfile.TemporaryDirectory() as tmpdir:
        backend = FileBasedBackend(fail_silently=fail_silently, file_path=tmpdir)
        result = backend.send_messages([])
        assert isinstance(result, int), f"Expected int, got {type(result).__name__}: {result}"
        assert result == 0, f"Expected 0 for empty messages, got {result}"
```

**Failing input**: `fail_silently=False` (or `True` - both fail)

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/path/to/django')

import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django.conf.global_settings')

from django.core.mail.backends.filebased import EmailBackend
import tempfile

with tempfile.TemporaryDirectory() as tmpdir:
    backend = EmailBackend(file_path=tmpdir)
    result = backend.send_messages([])

    print(f"Result: {result}")
    print(f"Type: {type(result)}")
    assert result == 0
```

## Why This Is A Bug

The base class documentation in `django.core.mail.backends.base.BaseEmailBackend.send_messages()` states:

```python
def send_messages(self, email_messages):
    """
    Send one or more EmailMessage objects and return the number of email
    messages sent.
    """
```

The filebased backend inherits from `ConsoleEmailBackend`, which has a bug where it returns `None` for empty messages (line 31 in `console.py`). All other backends (SMTP, dummy, locmem) correctly return `0` when given empty messages.

This inconsistency breaks code that relies on the documented return type being an integer.

## Fix

Since `FileBasedBackend` inherits from `ConsoleEmailBackend`, fixing the console backend (as described in the separate bug report for console.py) will also fix this issue. Alternatively, filebased can override the method:

```diff
--- a/django/core/mail/backends/console.py
+++ b/django/core/mail/backends/console.py
@@ -28,7 +28,7 @@ class EmailBackend(BaseEmailBackend):
     def send_messages(self, email_messages):
         """Write all messages to the stream in a thread-safe way."""
         if not email_messages:
-            return
+            return 0
         msg_count = 0
         with self._lock:
             try:
```