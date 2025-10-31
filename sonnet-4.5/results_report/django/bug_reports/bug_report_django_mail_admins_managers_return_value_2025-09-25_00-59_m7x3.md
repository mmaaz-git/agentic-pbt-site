# Bug Report: mail_admins and mail_managers Don't Return Send Result

**Target**: `django.core.mail.mail_admins` and `django.core.mail.mail_managers`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `mail_admins()` and `mail_managers()` functions don't return the result from `send()`, making them inconsistent with `send_mail()` and `send_mass_mail()` which return the number of messages sent. This prevents users from knowing if their message was successfully sent.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from django.core.mail import send_mail, mail_admins
from django.conf import settings

@given(st.text(min_size=1, max_size=100), st.text(min_size=1, max_size=200))
def test_email_functions_return_send_count(subject, message):
    """All email sending functions should return the count of messages sent"""

    # send_mail returns count
    result1 = send_mail(subject, message, 'from@example.com', ['to@example.com'])
    assert isinstance(result1, int) and result1 >= 0

    # mail_admins should also return count, but returns None
    settings.ADMINS = [('Admin', 'admin@example.com')]
    result2 = mail_admins(subject, message)
    assert isinstance(result2, int) and result2 >= 0, f"Expected int, got {type(result2)}"
```

**Failing input**: Any valid subject and message strings

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

send_result = send_mail('Test', 'Message', 'from@example.com', ['to@example.com'])
print(f"send_mail() returned: {send_result}")

admins_result = mail_admins('Test', 'Message')
print(f"mail_admins() returned: {admins_result}")

managers_result = mail_managers('Test', 'Message')
print(f"mail_managers() returned: {managers_result}")
```

**Output**:
```
send_mail() returned: 1
mail_admins() returned: None
mail_managers() returned: None
```

## Why This Is A Bug

This inconsistency violates the principle of least surprise and creates an inconsistent API:

1. **API inconsistency**: `send_mail()` and `send_mass_mail()` return message counts, but `mail_admins()` and `mail_managers()` return `None`

2. **Loss of information**: Users cannot determine if their admin/manager emails were actually sent without checking backend-specific state

3. **Inconsistent error handling**: With `send_mail()`, users can check `if result > 0` to verify success. This pattern doesn't work with `mail_admins()`

4. **Documentation mismatch**: The pattern established by other functions suggests all email functions should return send counts

## Fix

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