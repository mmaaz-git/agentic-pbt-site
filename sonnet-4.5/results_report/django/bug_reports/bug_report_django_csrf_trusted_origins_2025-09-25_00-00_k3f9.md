# Bug Report: Django CSRF_TRUSTED_ORIGINS Check Accepts Malformed URLs

**Target**: `django.core.checks.compatibility.django_4_0.check_csrf_trusted_origins`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `check_csrf_trusted_origins` function accepts malformed URLs that contain "://" but cannot be properly parsed by `urlsplit()`. This allows invalid CSRF_TRUSTED_ORIGINS configurations that the middleware cannot use correctly, leading to silent configuration errors.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st
from unittest.mock import patch
from urllib.parse import urlsplit

@given(st.builds(lambda prefix, suffix: prefix + "://" + suffix, st.text(), st.text()))
@settings(max_examples=10000)
def test_origins_that_pass_check_are_usable_by_middleware(origin):
    with patch('django.core.checks.compatibility.django_4_0.settings') as mock_settings:
        mock_settings.CSRF_TRUSTED_ORIGINS = [origin]
        errors = check_csrf_trusted_origins(None)

        if len(errors) == 0:
            parsed = urlsplit(origin)
            netloc = parsed.netloc.lstrip("*")

            assert parsed.scheme, f"Origin '{origin}' has no scheme but passed check"
            assert netloc, f"Origin '{origin}' has no netloc but passed check"
```

**Failing input**: `'://'`

## Reproducing the Bug

```python
from unittest.mock import patch
from urllib.parse import urlsplit
from django.core.checks.compatibility.django_4_0 import check_csrf_trusted_origins

malformed_origins = ["://", "://example.com", "http://"]

for origin in malformed_origins:
    with patch('django.core.checks.compatibility.django_4_0.settings') as mock_settings:
        mock_settings.CSRF_TRUSTED_ORIGINS = [origin]
        errors = check_csrf_trusted_origins(None)

        print(f"Origin: '{origin}'")
        print(f"  Passes check: {len(errors) == 0}")

        parsed = urlsplit(origin)
        print(f"  scheme: '{parsed.scheme}', netloc: '{parsed.netloc}'")
```

Output:
```
Origin: '://'
  Passes check: True
  scheme: '', netloc: ''
Origin: '://example.com'
  Passes check: True
  scheme: '', netloc: ''
Origin: 'http://'
  Passes check: True
  scheme: 'http', netloc: ''
```

## Why This Is A Bug

The CSRF middleware uses `urlsplit()` to parse CSRF_TRUSTED_ORIGINS and extract the scheme and netloc components (see `django/middleware/csrf.py:175-198`). When origins like "://" or "://example.com" are configured, they pass the check but produce empty schemes and netlocs when parsed.

This leads to:
1. Empty strings in `csrf_trusted_origins_hosts` list
2. Empty scheme keys in `allowed_origin_subdomains` mapping
3. Silent configuration errors where the CSRF protection doesn't work as intended

The check function should validate that origins are properly formatted URLs, not just that they contain "://".

## Fix

```diff
--- a/django/core/checks/compatibility/django_4_0.py
+++ b/django/core/checks/compatibility/django_4_0.py
@@ -1,4 +1,5 @@
 from django.conf import settings
+from urllib.parse import urlsplit

 from .. import Error, Tags, register

@@ -7,7 +8,15 @@ from .. import Error, Tags, register
 def check_csrf_trusted_origins(app_configs, **kwargs):
     errors = []
     for origin in settings.CSRF_TRUSTED_ORIGINS:
-        if "://" not in origin:
+        parsed = urlsplit(origin)
+        if not parsed.scheme or not parsed.netloc:
             errors.append(
                 Error(
-                    "As of Django 4.0, the values in the CSRF_TRUSTED_ORIGINS "
-                    "setting must start with a scheme (usually http:// or "
-                    "https://) but found %s. See the release notes for details."
+                    "The values in the CSRF_TRUSTED_ORIGINS setting must be "
+                    "valid URLs with both a scheme and netloc (e.g., "
+                    "http://example.com or https://*.example.com) but found %s. "
+                    "See the release notes for details."
                     % origin,
                     id="4_0.E001",
                 )
             )
     return errors
```