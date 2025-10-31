# Bug Report: django.core.mail.EmailMessage.attach() TypeError with None Filename

**Target**: `django.core.mail.EmailMessage.attach()`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `EmailMessage.attach()` method crashes with a `TypeError` when called with `filename=None`, despite the docstring explicitly stating "The filename can be omitted" and the method signature defaulting filename to None.

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

# Run the test
if __name__ == "__main__":
    test_attach_with_none_filename_and_mimetype()
```

<details>

<summary>
**Failing input**: `content=b''`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/36/hypo.py", line 27, in <module>
    test_attach_with_none_filename_and_mimetype()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/36/hypo.py", line 13, in test_attach_with_none_filename_and_mimetype
    @settings(max_examples=200)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/36/hypo.py", line 21, in test_attach_with_none_filename_and_mimetype
    msg.attach(filename=None, content=content, mimetype=None)
    ~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/django/core/mail/message.py", line 333, in attach
    or mimetypes.guess_type(filename)[0]
       ~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/mimetypes.py", line 322, in guess_type
    return _db.guess_type(url, strict)
           ~~~~~~~~~~~~~~^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/mimetypes.py", line 123, in guess_type
    url = os.fspath(url)
TypeError: expected str, bytes or os.PathLike object, not NoneType
Falsifying example: test_attach_with_none_filename_and_mimetype(
    content=b'',  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
from django.core.mail import EmailMessage
from django.conf import settings

# Configure Django settings
if not settings.configured:
    settings.configure(
        DEFAULT_CHARSET='utf-8',
        EMAIL_BACKEND='django.core.mail.backends.locmem.EmailBackend',
        DEFAULT_FROM_EMAIL='test@example.com',
    )

# Create an EmailMessage instance
msg = EmailMessage(
    subject="Test",
    body="Test",
    from_email="test@example.com",
    to=["to@example.com"],
)

# Try to attach with None filename and None mimetype
# According to docstring: "The filename can be omitted and the mimetype is guessed, if not provided."
try:
    msg.attach(filename=None, content=b"test content", mimetype=None)
    print("Success: Attachment added without error")
    message = msg.message()
    print("Success: Message created")
except TypeError as e:
    print(f"TypeError occurred: {e}")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"Other error occurred: {e}")
    import traceback
    traceback.print_exc()
```

<details>

<summary>
TypeError when attempting to attach with None filename
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/36/repo.py", line 23, in <module>
    msg.attach(filename=None, content=b"test content", mimetype=None)
    ~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/django/core/mail/message.py", line 333, in attach
    or mimetypes.guess_type(filename)[0]
       ~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/mimetypes.py", line 322, in guess_type
    return _db.guess_type(url, strict)
           ~~~~~~~~~~~~~~^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/mimetypes.py", line 123, in guess_type
    url = os.fspath(url)
TypeError: expected str, bytes or os.PathLike object, not NoneType
TypeError occurred: expected str, bytes or os.PathLike object, not NoneType
```
</details>

## Why This Is A Bug

This is a clear bug for multiple reasons:

1. **Docstring Contract Violation**: The docstring at line 311-312 explicitly states "The filename can be omitted and the mimetype is guessed, if not provided." This creates a documented API contract that the method should accept an omitted filename.

2. **Method Signature Implies None is Valid**: The method signature `def attach(self, filename=None, content=None, mimetype=None)` defaults filename to None, which in Python conventions strongly signals that None is an acceptable value.

3. **Implementation Oversight**: At line 333, the code calls `mimetypes.guess_type(filename)` without checking if filename is None first. Python's standard library `mimetypes.guess_type()` requires a string, bytes, or os.PathLike object and explicitly does not accept None.

4. **Internal Data Structure Supports None**: The EmailAttachment namedtuple (line 195) can technically store None as a filename value, showing that the internal representation doesn't prohibit this.

5. **Inconsistent Error Handling**: The method properly checks for `content is None` (line 328) but fails to handle `filename is None` despite it being a documented valid case.

## Relevant Context

The bug occurs in Django's email handling module at `/django/core/mail/message.py:333`. The method supports two distinct usage patterns:
1. Passing a MIMEBase instance directly (lines 321-327)
2. Passing filename, content, and mimetype parameters (lines 328-347)

The bug affects the second pattern. When filename is None and mimetype is also None, the code attempts to guess the mimetype from the filename, causing the crash. The logic should skip the guess_type call when filename is None and fall back to the DEFAULT_ATTACHMENT_MIME_TYPE.

Relevant source code link: [django/core/mail/message.py](https://github.com/django/django/blob/main/django/core/mail/message.py#L333)

Documentation reference: [Django Email Documentation](https://docs.djangoproject.com/en/stable/topics/email/)

## Proposed Fix

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