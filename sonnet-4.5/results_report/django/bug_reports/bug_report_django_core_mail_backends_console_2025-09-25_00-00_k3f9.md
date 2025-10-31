# Bug Report: django.core.mail.backends.console Returns None for Empty Messages

**Target**: `django.core.mail.backends.console.EmailBackend.send_messages`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `send_messages()` method in `ConsoleBackend` returns `None` instead of `0` when called with an empty list of messages, violating the documented contract that it should return the number of messages sent (an integer).

## Property-Based Test

```python
import io
from hypothesis import given, strategies as st
from django.core.mail.backends.console import EmailBackend as ConsoleBackend

@given(st.integers(min_value=0, max_value=10))
def test_console_backend_empty_list_returns_int(n):
    backend = ConsoleBackend(stream=io.StringIO())
    result = backend.send_messages([])
    assert isinstance(result, int), f"Expected int, got {type(result).__name__}: {result}"
```

**Failing input**: `[]` (empty list)

## Reproducing the Bug

```python
import io
from django.core.mail.backends.console import EmailBackend

backend = EmailBackend(stream=io.StringIO())
result = backend.send_messages([])

print(f"Result: {result}")
print(f"Type: {type(result).__name__}")
assert isinstance(result, int), f"Expected int, got {type(result)}"
assert result == 0, f"Expected 0, got {result}"
```

## Why This Is A Bug

The base class docstring at `django/core/mail/backends/base.py:55-59` states:

```python
def send_messages(self, email_messages):
    """
    Send one or more EmailMessage objects and return the number of email
    messages sent.
    """
```

All implementations should return an integer representing the count of messages sent. When no messages are provided, this should be `0`, not `None`. Other backends (smtp, locmem, dummy) correctly return `0` for empty message lists.

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