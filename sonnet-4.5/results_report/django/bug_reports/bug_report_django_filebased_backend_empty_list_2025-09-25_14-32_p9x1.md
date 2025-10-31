# Bug Report: FileBased Backend Returns None for Empty Message List

**Target**: `django.core.mail.backends.filebased.EmailBackend.send_messages`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The file-based email backend inherits from console backend and returns `None` instead of `0` when `send_messages()` is called with an empty message list, violating the API contract that specifies it should return the number of messages sent.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import tempfile
from django.core.mail.backends.filebased import EmailBackend

@given(st.integers(min_value=0, max_value=100))
@settings(max_examples=100)
def test_filebased_backend_returns_count(num_messages):
    with tempfile.TemporaryDirectory() as tmpdir:
        backend = EmailBackend(file_path=tmpdir, fail_silently=True)
        messages = [create_mock_message() for _ in range(num_messages)]

        result = backend.send_messages(messages)

        assert isinstance(result, int), f"Expected int, got {type(result)}"
        assert result >= 0, f"Expected non-negative count, got {result}"
```

**Failing input**: `num_messages=0`

## Reproducing the Bug

```python
import tempfile
from django.core.mail.backends.filebased import EmailBackend

with tempfile.TemporaryDirectory() as tmpdir:
    backend = EmailBackend(file_path=tmpdir)
    result = backend.send_messages([])

    assert result is None
    assert result != 0
```

## Why This Is A Bug

The base class `BaseEmailBackend` and all other backend implementations document that `send_messages()` should "return the number of email messages sent". When an empty list is provided, the method should return `0`, not `None`. This violates the API contract and breaks code that expects an integer return value.

Since `FileBasedBackend` inherits from `ConsoleEmailBackend`, it inherits this bug.

## Fix

The fix is the same as for the console backend (see bug_report_django_console_backend_empty_list_2025-09-25_14-30_k3m9.md):

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