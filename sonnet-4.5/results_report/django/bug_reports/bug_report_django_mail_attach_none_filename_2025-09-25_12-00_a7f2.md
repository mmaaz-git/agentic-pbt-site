# Bug Report: django.core.mail.EmailMessage.attach() Crashes with None Filename

**Target**: `django.core.mail.EmailMessage.attach()`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `EmailMessage.attach()` method crashes with a `TypeError` when `filename=None` and `mimetype=None`, despite the docstring stating "The filename can be omitted".

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st
from django.core.mail import EmailMessage
from django.conf import settings as django_settings

if not django_settings.configured:
    django_settings.configure(
        DEFAULT_CHARSET='utf-8',
        EMAIL_BACKEND='django.core.mail.backends.locmem.EmailBackend',
        DEFAULT_FROM_EMAIL='test@example.com',
    )

@given(content=st.binary(min_size=0, max_size=1000))
@settings(max_examples=200)
def test_attach_with_none_filename_and_mimetype(content):
    msg = EmailMessage(
        subject="Test",
        body="Test",
        from_email="test@example.com",
        to=["to@example.com"],
    )
    msg.attach(filename=None, content=content, mimetype=None)
    message = msg.message()
    assert message is not None
```

**Failing input**: `content=b''` (or any binary content)

## Reproducing the Bug

```python
from django.core.mail import EmailMessage
from django.conf import settings

if not settings.configured:
    settings.configure(
        DEFAULT_CHARSET='utf-8',
        EMAIL_BACKEND='django.core.mail.backends.locmem.EmailBackend',
        DEFAULT_FROM_EMAIL='test@example.com',
    )

msg = EmailMessage(
    subject="Test",
    body="Test",
    from_email="test@example.com",
    to=["to@example.com"],
)
msg.attach(filename=None, content=b"test content", mimetype=None)
```

## Why This Is A Bug

The docstring explicitly states: "The filename can be omitted and the mimetype is guessed, if not provided." This implies that `filename=None` should be valid. However, the code attempts to call `mimetypes.guess_type(None)`, which raises `TypeError: expected str, bytes or os.PathLike object, not NoneType`.

## Fix

```diff
--- a/django/core/mail/message.py
+++ b/django/core/mail/message.py
@@ -330,7 +330,7 @@ class EmailMessage:
         else:
             mimetype = (
                 mimetype
-                or mimetypes.guess_type(filename)[0]
+                or (mimetypes.guess_type(filename)[0] if filename else None)
                 or DEFAULT_ATTACHMENT_MIME_TYPE
             )
             basetype, subtype = mimetype.split("/", 1)
```