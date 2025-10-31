# Bug Report: django.core.checks.security.csrf check_csrf_failure_view Exception Handling

**Target**: `django.core.checks.security.csrf.check_csrf_failure_view`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `check_csrf_failure_view` function crashes with an unhandled `ViewDoesNotExist` exception when `CSRF_FAILURE_VIEW` points to a valid module but nonexistent view attribute, instead of returning an error message as intended.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
from django.core.checks.security.csrf import check_csrf_failure_view
from django.core.exceptions import ViewDoesNotExist

@given(st.text(min_size=1), st.text(min_size=1))
def test_csrf_failure_view_should_not_crash(module_name, view_name):
    assume('.' not in module_name)
    assume('.' not in view_name)

    settings.CSRF_FAILURE_VIEW = f'{module_name}.{view_name}'

    try:
        result = check_csrf_failure_view(None)
        assert isinstance(result, list), "Should return a list of errors"
    except ViewDoesNotExist:
        raise AssertionError(
            f"BUG: check_csrf_failure_view raised ViewDoesNotExist for "
            f"'{settings.CSRF_FAILURE_VIEW}' instead of returning error list"
        )
```

**Failing input**: `module_name='__main__', view_name='0'` → `CSRF_FAILURE_VIEW = '__main__.0'`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/path/to/django')
import os
os.environ['DJANGO_SETTINGS_MODULE'] = 'settings'

with open('settings.py', 'w') as f:
    f.write("""
SECRET_KEY = 'test-key-12345678901234567890123456789012345678901234567890'
MIDDLEWARE = ['django.middleware.csrf.CsrfViewMiddleware']
CSRF_FAILURE_VIEW = 'os.path.nonexistent_view'
""")

import django
django.setup()

from django.core.checks.security.csrf import check_csrf_failure_view

result = check_csrf_failure_view(None)
```

**Expected**: Returns `[Error(..., id='security.E102')]` indicating the view could not be imported.

**Actual**: Raises `django.core.exceptions.ViewDoesNotExist: Could not import 'os.path.nonexistent_view'. View does not exist in module os.path.`

## Why This Is A Bug

Security checks are designed to report configuration errors gracefully, not crash. The function is documented to catch import errors and return an `Error` object with id `'security.E102'`. However, when `_get_failure_view()` raises `ViewDoesNotExist` (which happens when the module exists but the view attribute doesn't), this exception is not caught, causing the check system to crash.

This is a legitimate user configuration error that should be reported via the check framework, not crash the application.

## Fix

```diff
--- a/django/core/checks/security/csrf.py
+++ b/django/core/checks/security/csrf.py
@@ -1,5 +1,6 @@
 import inspect

 from django.conf import settings
+from django.core.exceptions import ViewDoesNotExist

 from .. import Error, Tags, Warning, register

@@ -45,7 +46,7 @@ W016 = Warning(
 @register(Tags.security)
 def check_csrf_failure_view(app_configs, **kwargs):
     from django.middleware.csrf import _get_failure_view

     errors = []
     try:
         view = _get_failure_view()
-    except ImportError:
+    except (ImportError, ViewDoesNotExist):
         msg = (
             "The CSRF failure view '%s' could not be imported."
             % settings.CSRF_FAILURE_VIEW
         )
         errors.append(Error(msg, id="security.E102"))
```