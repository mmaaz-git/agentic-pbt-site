# Bug Report: django.core.mail.backends.console Returns None for Empty Message List

**Target**: `django.core.mail.backends.console.EmailBackend.send_messages`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `send_messages` method in `console.EmailBackend` returns `None` instead of `0` when called with an empty list of messages, violating the API contract specified in the base class.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

import io
from hypothesis import given, strategies as st
from django.core.mail.backends.console import EmailBackend as ConsoleBackend
from unittest.mock import Mock

@given(st.lists(st.just(Mock(message=lambda: Mock(as_bytes=lambda: b'test', get_charset=lambda: None))), max_size=10))
def test_console_backend_return_type_invariant(messages):
    stream = io.StringIO()
    backend = ConsoleBackend(stream=stream, fail_silently=False)
    result = backend.send_messages(messages)

    assert result is not None, f"send_messages should return an integer, not None"
    assert isinstance(result, int), f"send_messages should return an integer"
    assert result == len(messages), f"send_messages should return message count"
```

**Failing input**: `[]` (empty list)

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

import io
from django.core.mail.backends.console import EmailBackend

stream = io.StringIO()
backend = EmailBackend(stream=stream)
result = backend.send_messages([])

print(f"Result: {result!r}")
print(f"Expected: 0")

assert result == 0
```

## Why This Is A Bug

The `BaseEmailBackend.send_messages` docstring explicitly states:

> Send one or more EmailMessage objects and return the number of email messages sent.

The SMTP backend correctly returns `0` for empty message lists (smtp.py:125-126), but the console backend returns `None` (console.py:30-31). This violates the documented API contract and creates inconsistent behavior across backends. Since `filebased.EmailBackend` inherits from `console.EmailBackend` without overriding `send_messages`, it has the same bug.

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