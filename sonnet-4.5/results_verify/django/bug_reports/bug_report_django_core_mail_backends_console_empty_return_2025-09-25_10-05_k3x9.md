# Bug Report: django.core.mail.backends.console Returns None Instead of 0

**Target**: `django.core.mail.backends.console.EmailBackend.send_messages`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `console.EmailBackend.send_messages` method returns `None` when given an empty list of messages, violating the documented API contract that states it should "return the number of email messages sent".

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
import io
from django.core.mail.backends.console import EmailBackend

@given(st.lists(st.text()))
def test_console_backend_empty_messages_returns_count(message_bodies):
    assume(len(message_bodies) == 0)

    stream = io.StringIO()
    backend = EmailBackend(stream=stream)

    messages = []
    result = backend.send_messages(messages)

    assert result is not None, "send_messages should return a count, not None"
    assert result == 0, f"Expected 0 messages sent, got {result}"
```

**Failing input**: `message_bodies=[]` (empty list)

## Reproducing the Bug

```python
import io
import django
from django.conf import settings

if not settings.configured:
    settings.configure(DEBUG=True, SECRET_KEY='test', DEFAULT_CHARSET='utf-8')
    django.setup()

from django.core.mail.backends.console import EmailBackend

stream = io.StringIO()
backend = EmailBackend(stream=stream)

result = backend.send_messages([])

print(f"Result: {result}")
print(f"Expected: 0")
print(f"Bug: Returns None instead of 0")
```

## Why This Is A Bug

The `BaseEmailBackend.send_messages` docstring (line 56-58 in base.py) states:

> "Send one or more EmailMessage objects and return the number of email messages sent."

The console backend should return `0` when no messages are sent, not `None`. This violates the API contract and causes type inconsistency - callers expecting an integer may receive `None` instead.

This bug also affects `filebased.EmailBackend` since it inherits from `console.EmailBackend`.

Other backends (dummy, locmem, smtp) correctly handle this case:
- `dummy.py:10` returns `len(list(email_messages))` which is 0 for empty lists
- `locmem.py:28` initializes `msg_count = 0` and returns it
- `smtp.py:126` returns `0` for empty messages

## Fix

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