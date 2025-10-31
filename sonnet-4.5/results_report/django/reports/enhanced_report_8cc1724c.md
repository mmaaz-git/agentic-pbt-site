# Bug Report: Django mail_admins and mail_managers Return None Instead of Message Count

**Target**: `django.core.mail.mail_admins` and `django.core.mail.mail_managers`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `mail_admins()` and `mail_managers()` functions return `None` instead of returning the number of messages sent, making them inconsistent with `send_mail()` and `send_mass_mail()` which return message counts.

## Property-Based Test

```python
import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'settings')

import django
from django.conf import settings

if not settings.configured:
    settings.configure(
        SECRET_KEY='test-secret-key',
        EMAIL_BACKEND='django.core.mail.backends.locmem.EmailBackend',
        DEFAULT_FROM_EMAIL='test@example.com',
        EMAIL_SUBJECT_PREFIX='[Django] ',
        SERVER_EMAIL='root@localhost',
        ADMINS=[('Admin', 'admin@example.com')],
        MANAGERS=[('Manager', 'manager@example.com')],
    )

django.setup()

from hypothesis import given, strategies as st, assume
from django.core.mail import send_mail, mail_admins

@given(
    st.text(min_size=1, max_size=100).filter(lambda x: '\n' not in x and '\r' not in x),
    st.text(min_size=1, max_size=200)
)
def test_email_functions_return_send_count(subject, message):
    """All email sending functions should return the count of messages sent"""

    # send_mail returns count
    result1 = send_mail(subject, message, 'from@example.com', ['to@example.com'])
    assert isinstance(result1, int) and result1 >= 0

    # mail_admins should also return count, but returns None
    settings.ADMINS = [('Admin', 'admin@example.com')]
    result2 = mail_admins(subject, message)
    assert isinstance(result2, int) and result2 >= 0, f"Expected int, got {type(result2)}"

# Run the test
test_email_functions_return_send_count()
```

<details>

<summary>
**Failing input**: `subject='0', message='0'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/16/hypo.py", line 40, in <module>
    test_email_functions_return_send_count()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/16/hypo.py", line 24, in test_email_functions_return_send_count
    st.text(min_size=1, max_size=100).filter(lambda x: '\n' not in x and '\r' not in x),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/16/hypo.py", line 37, in test_email_functions_return_send_count
    assert isinstance(result2, int) and result2 >= 0, f"Expected int, got {type(result2)}"
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Expected int, got <class 'NoneType'>
Falsifying example: test_email_functions_return_send_count(
    # The test always failed when commented parts were varied together.
    subject='0',  # or any other generated value
    message='0',  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'settings')

import django
from django.conf import settings

if not settings.configured:
    settings.configure(
        SECRET_KEY='test-secret-key',
        EMAIL_BACKEND='django.core.mail.backends.locmem.EmailBackend',
        DEFAULT_FROM_EMAIL='test@example.com',
        EMAIL_SUBJECT_PREFIX='[Django] ',
        SERVER_EMAIL='root@localhost',
        ADMINS=[('Admin', 'admin@example.com')],
        MANAGERS=[('Manager', 'manager@example.com')],
    )

django.setup()

from django.core.mail import send_mail, mail_admins, mail_managers

# Test send_mail - should return 1
send_result = send_mail('Test Subject', 'Test Message', 'from@example.com', ['to@example.com'])
print(f"send_mail() returned: {send_result} (type: {type(send_result)})")

# Test mail_admins - should return 1 but returns None
admins_result = mail_admins('Admin Alert', 'Important message for admins')
print(f"mail_admins() returned: {admins_result} (type: {type(admins_result)})")

# Test mail_managers - should return 1 but returns None
managers_result = mail_managers('Manager Notice', 'Important message for managers')
print(f"mail_managers() returned: {managers_result} (type: {type(managers_result)})")
```

<details>

<summary>
Output showing inconsistent return values
</summary>
```
send_mail() returned: 1 (type: <class 'int'>)
mail_admins() returned: None (type: <class 'NoneType'>)
mail_managers() returned: None (type: <class 'NoneType'>)
```
</details>

## Why This Is A Bug

This is a bug because it creates an inconsistent API within Django's email module. All four email-sending functions in `django.core.mail` (`send_mail`, `send_mass_mail`, `mail_admins`, `mail_managers`) ultimately call the same underlying `mail.send()` method which returns an integer count of messages sent. However, only two of these functions pass that return value back to the caller.

This inconsistency violates the principle of least surprise. Users who are familiar with `send_mail()` returning a count (as documented) would reasonably expect similar functions in the same module to behave consistently. The current behavior prevents users from:

1. Programmatically verifying that admin/manager emails were sent
2. Using consistent error handling patterns across email functions
3. Writing generic email handling code that works with all Django email functions

The Django documentation explicitly states that `send_mail()` returns "the number of successfully delivered messages" but doesn't document any return value for `mail_admins()` and `mail_managers()`. This documentation gap, combined with the actual code behavior, suggests an oversight rather than intentional design.

## Relevant Context

The issue is clearly visible in Django's source code at `/django/core/mail/__init__.py`:

- Line 92: `send_mail()` ends with `return mail.send()`
- Line 119: `send_mass_mail()` ends with `return connection.send_messages(messages)`
- Line 139: `mail_admins()` calls `mail.send(fail_silently=fail_silently)` without returning the result
- Line 159: `mail_managers()` calls `mail.send(fail_silently=fail_silently)` without returning the result

The `mail.send()` method does return an integer (the number of messages sent), as evidenced by `send_mail()` returning this value. The omission of the `return` statement in `mail_admins()` and `mail_managers()` appears to be an oversight.

Django documentation for email functions: https://docs.djangoproject.com/en/5.0/topics/email/

## Proposed Fix

```diff
--- a/django/core/mail/__init__.py
+++ b/django/core/mail/__init__.py
@@ -136,7 +136,7 @@ def mail_admins(
     )
     if html_message:
         mail.attach_alternative(html_message, "text/html")
-    mail.send(fail_silently=fail_silently)
+    return mail.send(fail_silently=fail_silently)


 def mail_managers(
@@ -156,4 +156,4 @@ def mail_managers(
     )
     if html_message:
         mail.attach_alternative(html_message, "text/html")
-    mail.send(fail_silently=fail_silently)
+    return mail.send(fail_silently=fail_silently)
```