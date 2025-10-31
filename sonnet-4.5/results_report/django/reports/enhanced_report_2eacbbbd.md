# Bug Report: Django Mail Filebased Backend Returns None Instead of 0 for Empty Message Lists

**Target**: `django.core.mail.backends.filebased.EmailBackend` and `django.core.mail.backends.console.EmailBackend`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The filebased and console email backends violate the documented API contract by returning `None` instead of `0` when `send_messages()` is called with an empty list, causing type errors and breaking backend substitutability.

## Property-Based Test

```python
#!/usr/bin/env python3
"""Property-based test for Django email backend empty messages bug."""

import tempfile
import django
from django.conf import settings
from hypothesis import given, strategies as st

# Configure Django settings
settings.configure(
    EMAIL_HOST='localhost',
    EMAIL_PORT=25,
    EMAIL_HOST_USER='',
    EMAIL_HOST_PASSWORD='',
    EMAIL_USE_TLS=False,
    EMAIL_USE_SSL=False,
    DEFAULT_CHARSET='utf-8',
)
django.setup()

from django.core.mail.backends.filebased import EmailBackend as FileBasedBackend
from django.core.mail.backends.smtp import EmailBackend as SMTPBackend
from django.core.mail.backends.locmem import EmailBackend as LocmemBackend

@given(st.just([]))
def test_empty_message_consistency_filebased(empty_messages):
    """Test that all email backends return consistent values for empty message lists."""
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

if __name__ == "__main__":
    test_empty_message_consistency_filebased()
```

<details>

<summary>
**Failing input**: `[]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/53/hypo.py", line 43, in <module>
    test_empty_message_consistency_filebased()
  File "/home/npc/pbt/agentic-pbt/worker_/53/hypo.py", line 26, in test_empty_message_consistency_filebased
    def test_empty_message_consistency_filebased(empty_messages):
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/53/hypo.py", line 37, in test_empty_message_consistency_filebased
    assert isinstance(filebased_result, int), \
            f"Expected int, got {type(filebased_result).__name__}"
AssertionError: Expected int, got NoneType
Falsifying example: test_empty_message_consistency_filebased(
    empty_messages=[],
)
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""Minimal reproduction of Django email backend empty messages bug."""

import tempfile
import django
from django.conf import settings

# Configure Django settings
settings.configure(
    EMAIL_HOST='localhost',
    EMAIL_PORT=25,
    EMAIL_HOST_USER='',
    EMAIL_HOST_PASSWORD='',
    EMAIL_USE_TLS=False,
    EMAIL_USE_SSL=False,
    DEFAULT_CHARSET='utf-8',
)
django.setup()

from django.core.mail.backends.filebased import EmailBackend as FileBasedBackend
from django.core.mail.backends.smtp import EmailBackend as SMTPBackend
from django.core.mail.backends.locmem import EmailBackend as LocmemBackend
from django.core.mail.backends.console import EmailBackend as ConsoleBackend

# Test with empty message list
empty_messages = []

# Create temporary directory for file-based backend
with tempfile.TemporaryDirectory() as tmpdir:
    # Initialize backends
    filebased_backend = FileBasedBackend(file_path=tmpdir)
    smtp_backend = SMTPBackend()
    locmem_backend = LocmemBackend()
    console_backend = ConsoleBackend()

    # Call send_messages with empty list
    filebased_result = filebased_backend.send_messages(empty_messages)
    smtp_result = smtp_backend.send_messages(empty_messages)
    locmem_result = locmem_backend.send_messages(empty_messages)
    console_result = console_backend.send_messages(empty_messages)

    # Display results
    print(f"Filebased backend result: {filebased_result!r} (type: {type(filebased_result).__name__})")
    print(f"Console backend result: {console_result!r} (type: {type(console_result).__name__})")
    print(f"SMTP backend result: {smtp_result!r} (type: {type(smtp_result).__name__})")
    print(f"Locmem backend result: {locmem_result!r} (type: {type(locmem_result).__name__})")

    print("\n--- Type inconsistency ---")
    print(f"Filebased returns: {type(filebased_result)}")
    print(f"Console returns: {type(console_result)}")
    print(f"SMTP returns: {type(smtp_result)}")
    print(f"Locmem returns: {type(locmem_result)}")

    print("\n--- Attempting numeric comparison ---")
    try:
        result = filebased_result > 0
        print(f"filebased_result > 0 = {result}")
    except TypeError as e:
        print(f"TypeError when comparing filebased_result > 0: {e}")

    try:
        result = console_result > 0
        print(f"console_result > 0 = {result}")
    except TypeError as e:
        print(f"TypeError when comparing console_result > 0: {e}")

    print("\n--- Backend substitutability broken ---")
    print("Cannot transparently swap filebased/console with smtp/locmem backends")
    print("due to different return types for empty message lists.")
```

<details>

<summary>
TypeError: '>' not supported between NoneType and int
</summary>
```
Filebased backend result: None (type: NoneType)
Console backend result: None (type: NoneType)
SMTP backend result: 0 (type: int)
Locmem backend result: 0 (type: int)

--- Type inconsistency ---
Filebased returns: <class 'NoneType'>
Console returns: <class 'NoneType'>
SMTP returns: <class 'int'>
Locmem returns: <class 'int'>

--- Attempting numeric comparison ---
TypeError when comparing filebased_result > 0: '>' not supported between instances of 'NoneType' and 'int'
TypeError when comparing console_result > 0: '>' not supported between instances of 'NoneType' and 'int'

--- Backend substitutability broken ---
Cannot transparently swap filebased/console with smtp/locmem backends
due to different return types for empty message lists.
```
</details>

## Why This Is A Bug

The `send_messages()` method in `BaseEmailBackend` explicitly documents in its docstring (lines 57-58 of `django/core/mail/backends/base.py`) that it should "Send one or more EmailMessage objects and return the number of email messages sent." For an empty list, the number of messages sent is mathematically 0, not None.

The bug occurs in `django/core/mail/backends/console.py` at lines 30-31:
```python
if not email_messages:
    return  # Returns None instead of 0
```

Since `filebased.EmailBackend` inherits from `console.EmailBackend` (line 11 of `filebased.py`), it inherits this buggy behavior. Meanwhile, the SMTP backend correctly returns 0 (line 126 of `smtp.py`), and locmem also returns 0 through its loop logic.

This violates three key principles:
1. **Type Contract**: The method must return an integer count per documentation, not NoneType
2. **Backend Substitutability**: All email backends should be interchangeable without changing behavior
3. **Python Conventions**: Counting operations should return integers (similar to `len()`, `count()`, etc.)

## Relevant Context

This bug affects production deployments where code relies on the return value of `send_messages()` for logging, metrics, or conditional logic. For example:
- Code that tracks email send statistics will fail with TypeError
- Unit tests that swap backends for testing will behave differently
- Monitoring systems that check if `send_messages() > 0` will crash

The filebased backend is commonly used in development/staging environments, while console backend is used for debugging. Having them behave differently from production backends (SMTP) makes testing unreliable.

Documentation references:
- BaseEmailBackend source: `/django/core/mail/backends/base.py:57-58`
- Console backend bug: `/django/core/mail/backends/console.py:30-31`
- Filebased inheritance: `/django/core/mail/backends/filebased.py:11`
- SMTP correct implementation: `/django/core/mail/backends/smtp.py:125-126`

## Proposed Fix

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