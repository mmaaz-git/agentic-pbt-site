# Bug Report: Django CSRF Trusted Origins Scheme Validation

**Target**: `django.core.checks.compatibility.django_4_0.check_csrf_trusted_origins`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `check_csrf_trusted_origins` function incorrectly validates CSRF_TRUSTED_ORIGINS by only checking if `"://"` appears anywhere in the origin string, rather than verifying that a non-empty scheme appears at the start. This allows malformed origins like `"://example.com"` to pass validation despite not starting with a valid scheme.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from unittest.mock import patch
from django.core.checks.compatibility.django_4_0 import check_csrf_trusted_origins

@given(st.text())
def test_scheme_must_be_at_start(origin):
    with patch('django.conf.settings.CSRF_TRUSTED_ORIGINS', [origin]):
        errors = check_csrf_trusted_origins(app_configs=None)

        if '://' in origin:
            index_of_separator = origin.index('://')
            if index_of_separator == 0:
                assert len(errors) > 0, \
                    f"Origin '{origin}' has no scheme before ://, should fail"
        else:
            assert len(errors) > 0, \
                f"Origin '{origin}' has no ://, should fail"
```

**Failing input**: `"://example.com"`

## Reproducing the Bug

```python
import os
os.environ['DJANGO_SETTINGS_MODULE'] = 'test_settings'

from django.conf import settings
if not settings.configured:
    settings.configure(
        SECRET_KEY='test',
        CSRF_TRUSTED_ORIGINS=[],
        SILENCED_SYSTEM_CHECKS=[],
    )

import django
django.setup()

from unittest.mock import patch
from django.core.checks.compatibility.django_4_0 import check_csrf_trusted_origins

with patch('django.conf.settings.CSRF_TRUSTED_ORIGINS', ['://example.com']):
    errors = check_csrf_trusted_origins(app_configs=None)
    print(f"Errors for '://example.com': {len(errors)}")

with patch('django.conf.settings.CSRF_TRUSTED_ORIGINS', ['example.com://foo']):
    errors = check_csrf_trusted_origins(app_configs=None)
    print(f"Errors for 'example.com://foo': {len(errors)}")
```

## Why This Is A Bug

The error message in the code explicitly states: "the values in the CSRF_TRUSTED_ORIGINS setting must **start with a scheme** (usually http:// or https://)". However, the validation logic only checks `if "://" not in origin`, which allows:

1. `"://example.com"` - contains "://" but has no scheme
2. `"example.com://foo"` - contains "://" but doesn't start with a scheme
3. Any string with "://" in the middle

According to Django's documentation, CSRF_TRUSTED_ORIGINS should contain properly formatted origin URLs like `"https://example.com"`. The current check fails to enforce this documented requirement.

## Fix

```diff
--- a/django/core/checks/compatibility/django_4_0.py
+++ b/django/core/checks/compatibility/django_4_0.py
@@ -7,7 +7,11 @@ from .. import Error, Tags, register
 def check_csrf_trusted_origins(app_configs, **kwargs):
     errors = []
     for origin in settings.CSRF_TRUSTED_ORIGINS:
-        if "://" not in origin:
+        if "://" not in origin or origin.index("://") == 0:
             errors.append(
                 Error(
                     "As of Django 4.0, the values in the CSRF_TRUSTED_ORIGINS "
```