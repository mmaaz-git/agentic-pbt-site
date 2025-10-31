# Bug Report: django.core.mail.backends.console Returns None Instead of Int

**Target**: `django.core.mail.backends.console.EmailBackend.send_messages`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `console.EmailBackend.send_messages()` method returns `None` instead of an integer when called with an empty list, violating the base class contract that specifies it should return the number of messages sent.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from django.core.mail.backends.console import EmailBackend

@given(st.lists(st.builds(lambda: None)))
def test_console_send_messages_returns_int(messages):
    backend = EmailBackend()
    result = backend.send_messages(messages)
    assert isinstance(result, int), f"Expected int, got {type(result).__name__}: {result}"
```

**Failing input**: `messages=[]`

## Reproducing the Bug

```python
from django.core.mail.backends.console import EmailBackend

backend = EmailBackend()
result = backend.send_messages([])
print(f"Result: {result}, Type: {type(result)}")
```

Output:
```
Result: None, Type: <class 'NoneType'>
```

Expected: `Result: 0, Type: <class 'int'>`

## Why This Is A Bug

The base class `BaseEmailBackend` documents that `send_messages()` should "return the number of email messages sent" (line 57-58 in base.py). All other backend implementations (smtp, dummy, locmem, filebased) correctly return an integer. The console backend returns `None` when the message list is empty, breaking this contract.

This violates the Liskov Substitution Principle - code that works with other backends will fail with the console backend when passing an empty list.

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