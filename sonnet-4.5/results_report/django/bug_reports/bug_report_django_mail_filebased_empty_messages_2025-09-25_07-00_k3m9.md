# Bug Report: Django Mail Filebased Backend Empty Messages Returns None

**Target**: `django.core.mail.backends.filebased.EmailBackend`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The filebased email backend returns `None` for empty message lists instead of `0`, inheriting this bug from the console backend and violating the backend substitutability contract.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from django.core.mail.backends.filebased import EmailBackend as FileBasedBackend
from django.core.mail.backends.smtp import EmailBackend as SMTPBackend
from django.core.mail.backends.locmem import EmailBackend as LocmemBackend
import tempfile

@given(st.just([]))
def test_empty_message_consistency_filebased(empty_messages):
    with tempfile.TemporaryDirectory() as tmpdir:
        filebased = FileBasedBackend(file_path=tmpdir)
        smtp = SMTPBackend()
        locmem = LocmemBackend()

        filebased_result = filebased.send_messages(empty_messages)
        smtp_result = smtp.send_messages(empty_messages)
        locmem_result = locmem.send_messages(empty_messages)

        assert isinstance(filebased_result, int), \
            f"Expected int, got {type(filebased_result).__name__}"
        assert filebased_result == smtp_result == locmem_result, \
            f"Inconsistent return values: filebased={filebased_result}, smtp={smtp_result}, locmem={locmem_result}"
```

**Failing input**: `[]` (empty list)

## Reproducing the Bug

```python
from django.core.mail.backends.filebased import EmailBackend
from django.core.mail.backends.smtp import EmailBackend as SMTPBackend
import tempfile

with tempfile.TemporaryDirectory() as tmpdir:
    filebased_backend = FileBasedBackend(file_path=tmpdir)
    smtp_backend = SMTPBackend()

    filebased_result = filebased_backend.send_messages([])
    smtp_result = smtp_backend.send_messages([])

    print(f"Filebased result: {filebased_result!r}")  # None
    print(f"SMTP result: {smtp_result!r}")            # 0
    print(f"Type mismatch: {type(filebased_result)} vs {type(smtp_result)}")

    result = filebased_result > 0  # TypeError: '>' not supported
```

## Why This Is A Bug

The filebased backend inherits from `console.EmailBackend` (line 11 of `filebased.py`):

```python
class EmailBackend(ConsoleEmailBackend):
```

The console backend has a bug at lines 30-31:

```python
def send_messages(self, email_messages):
    if not email_messages:
        return  # Returns None instead of 0
```

This means the filebased backend inherits the same bug. The base class docstring says `send_messages()` should "return the number of email messages sent", which should be `0` for empty lists, not `None`.

This breaks:
1. **Type consistency**: Returns `NoneType` instead of `int`
2. **Backend substitutability**: Cannot swap filebased with other backends
3. **API contract**: Violates the documented return type

## Fix

The fix requires modifying the parent class `console.EmailBackend`:

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
```

This will fix the bug for both console and filebased backends.