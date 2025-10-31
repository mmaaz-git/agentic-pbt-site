# Bug Report: Django CSRF_TRUSTED_ORIGINS Validation Accepts Malformed URLs

**Target**: `django.core.checks.compatibility.django_4_0.check_csrf_trusted_origins`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `check_csrf_trusted_origins` function incorrectly accepts malformed URLs that contain "://" but lack valid scheme and/or netloc components when parsed by `urlsplit()`. This causes silent configuration failures where CSRF protection doesn't work as intended.

## Property-Based Test

```python
#!/usr/bin/env python3
"""
Property-based test for Django CSRF_TRUSTED_ORIGINS validation bug.
This test verifies that origins passing the check can be used by the middleware.
"""

import sys
import os
# Add Django to path
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

# Configure Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django.conf.global_settings')
import django
from django.conf import settings
settings.configure(
    DEBUG=True,
    SECRET_KEY='test-secret-key',
    CSRF_TRUSTED_ORIGINS=[]
)

from hypothesis import given, settings as hsettings, strategies as st
from unittest.mock import patch
from urllib.parse import urlsplit
from django.core.checks.compatibility.django_4_0 import check_csrf_trusted_origins

@given(st.builds(lambda prefix, suffix: prefix + "://" + suffix, st.text(), st.text()))
@hsettings(max_examples=10000)
def test_origins_that_pass_check_are_usable_by_middleware(origin):
    with patch.object(settings, 'CSRF_TRUSTED_ORIGINS', [origin]):
        errors = check_csrf_trusted_origins(None)

        if len(errors) == 0:
            # If the origin passes the check, it should be usable by middleware
            parsed = urlsplit(origin)
            netloc = parsed.netloc.lstrip("*")

            assert parsed.scheme, f"Origin '{origin}' has no scheme but passed check"
            assert netloc, f"Origin '{origin}' has no netloc but passed check"

if __name__ == "__main__":
    print("Running property-based test for Django CSRF_TRUSTED_ORIGINS validation...")
    print("=" * 60)
    try:
        test_origins_that_pass_check_are_usable_by_middleware()
        print("Test passed!")
    except AssertionError as e:
        print(f"Test failed with assertion error: {e}")
        print("\nThis demonstrates that the check function accepts invalid origins.")
    except Exception as e:
        print(f"Test failed with error: {e}")
```

<details>

<summary>
**Failing input**: `'://'`
</summary>
```
Running property-based test for Django CSRF_TRUSTED_ORIGINS validation...
============================================================
Test failed with assertion error: Origin '://' has no scheme but passed check

This demonstrates that the check function accepts invalid origins.
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""
Minimal reproduction of Django CSRF_TRUSTED_ORIGINS validation bug.
This demonstrates that malformed URLs pass the check but produce
empty components when parsed, breaking the CSRF middleware.
"""

import sys
import os
# Add Django to path
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

# Configure Django settings first
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django.conf.global_settings')
import django
from django.conf import settings
settings.configure(
    DEBUG=True,
    SECRET_KEY='test-secret-key',
    CSRF_TRUSTED_ORIGINS=[]
)

from unittest.mock import patch
from urllib.parse import urlsplit

# Import the check function
from django.core.checks.compatibility.django_4_0 import check_csrf_trusted_origins

# Test cases of malformed URLs that should fail validation
malformed_origins = [
    "://",
    "://example.com",
    "http://",
    "https://"
]

print("Django CSRF_TRUSTED_ORIGINS Validation Bug Demonstration")
print("=" * 60)

for origin in malformed_origins:
    print(f"\nTesting origin: '{origin}'")
    print("-" * 40)

    # Test the check function
    with patch.object(settings, 'CSRF_TRUSTED_ORIGINS', [origin]):
        errors = check_csrf_trusted_origins(None)

        passes_check = len(errors) == 0
        print(f"  Passes validation check: {passes_check}")

        if errors:
            print(f"  Error message: {errors[0].msg}")

    # Show what urlsplit actually produces
    parsed = urlsplit(origin)
    print(f"  urlsplit() results:")
    print(f"    scheme: '{parsed.scheme}' (empty={not parsed.scheme})")
    print(f"    netloc: '{parsed.netloc}' (empty={not parsed.netloc})")

    # Show the impact on middleware
    if passes_check:
        print(f"  PROBLEM: This malformed URL passes validation but:")
        if not parsed.scheme:
            print(f"    - Has no scheme (middleware needs this)")
        if not parsed.netloc:
            print(f"    - Has no netloc (middleware needs this)")
        print(f"    - Would cause silent CSRF protection failure")

print("\n" + "=" * 60)
print("CONCLUSION: The check function incorrectly accepts malformed URLs")
print("that cannot be properly parsed by the CSRF middleware.")
```

<details>

<summary>
Malformed URLs pass validation but have empty components
</summary>
```
Django CSRF_TRUSTED_ORIGINS Validation Bug Demonstration
============================================================

Testing origin: '://'
----------------------------------------
  Passes validation check: True
  urlsplit() results:
    scheme: '' (empty=True)
    netloc: '' (empty=True)
  PROBLEM: This malformed URL passes validation but:
    - Has no scheme (middleware needs this)
    - Has no netloc (middleware needs this)
    - Would cause silent CSRF protection failure

Testing origin: '://example.com'
----------------------------------------
  Passes validation check: True
  urlsplit() results:
    scheme: '' (empty=True)
    netloc: '' (empty=True)
  PROBLEM: This malformed URL passes validation but:
    - Has no scheme (middleware needs this)
    - Has no netloc (middleware needs this)
    - Would cause silent CSRF protection failure

Testing origin: 'http://'
----------------------------------------
  Passes validation check: True
  urlsplit() results:
    scheme: 'http' (empty=False)
    netloc: '' (empty=True)
  PROBLEM: This malformed URL passes validation but:
    - Has no netloc (middleware needs this)
    - Would cause silent CSRF protection failure

Testing origin: 'https://'
----------------------------------------
  Passes validation check: True
  urlsplit() results:
    scheme: 'https' (empty=False)
    netloc: '' (empty=True)
  PROBLEM: This malformed URL passes validation but:
    - Has no netloc (middleware needs this)
    - Would cause silent CSRF protection failure

============================================================
CONCLUSION: The check function incorrectly accepts malformed URLs
that cannot be properly parsed by the CSRF middleware.
```
</details>

## Why This Is A Bug

The `check_csrf_trusted_origins` function is designed to validate CSRF_TRUSTED_ORIGINS configuration values as part of Django 4.0's compatibility checks. Its purpose is to ensure that configured origins can be properly used by the CSRF middleware for security validation.

The current implementation uses a naive string search for "://" (line 10 in django_4_0.py), which is insufficient because:

1. **Contract Violation**: The check function promises that passing values are valid for the middleware, but the middleware requires both `scheme` and `netloc` components from `urlsplit()` to function properly.

2. **Silent Failures**: When malformed URLs like "://" or "://example.com" are configured, they pass validation but produce empty scheme/netloc values. The middleware then:
   - Adds empty strings to `csrf_trusted_origins_hosts` list (line 177)
   - Creates invalid entries in `allowed_origin_subdomains` mapping (line 197)
   - Fails to provide CSRF protection for these origins without any error

3. **Documentation Mismatch**: The error message states values "must start with a scheme" but the check only verifies the presence of "://" anywhere in the string, not that a valid scheme exists.

4. **Security Impact**: This affects CSRF protection, a critical security feature. While it requires developer misconfiguration, validation checks exist specifically to catch such errors.

## Relevant Context

The CSRF middleware implementation (django/middleware/csrf.py:175-198) shows clear dependency on `urlsplit()` for parsing origins:

- Line 177: `urlsplit(origin).netloc.lstrip("*")` - expects valid netloc
- Line 193-197: Uses `parsed.scheme` and `parsed.netloc` for subdomain matching

The Django 4.0 release notes explicitly state that CSRF_TRUSTED_ORIGINS values must include a scheme, indicating this is a required component for proper functionality.

Documentation: https://docs.djangoproject.com/en/4.0/ref/settings/#csrf-trusted-origins

## Proposed Fix

```diff
--- a/django/core/checks/compatibility/django_4_0.py
+++ b/django/core/checks/compatibility/django_4_0.py
@@ -1,4 +1,5 @@
 from django.conf import settings
+from urllib.parse import urlsplit

 from .. import Error, Tags, register

@@ -7,13 +8,20 @@
 def check_csrf_trusted_origins(app_configs, **kwargs):
     errors = []
     for origin in settings.CSRF_TRUSTED_ORIGINS:
-        if "://" not in origin:
+        try:
+            parsed = urlsplit(origin)
+            if not parsed.scheme or not parsed.netloc:
+                errors.append(
+                    Error(
+                        "As of Django 4.0, the values in the CSRF_TRUSTED_ORIGINS "
+                        "setting must be valid URLs with both a scheme and netloc "
+                        "(e.g., 'https://example.com' or 'https://*.example.com') "
+                        "but found '%s'. See the release notes for details." % origin,
+                        id="4_0.E001",
+                    )
+                )
+        except Exception:
             errors.append(
                 Error(
-                    "As of Django 4.0, the values in the CSRF_TRUSTED_ORIGINS "
-                    "setting must start with a scheme (usually http:// or "
-                    "https://) but found %s. See the release notes for details."
-                    % origin,
+                    "Invalid URL format in CSRF_TRUSTED_ORIGINS: '%s'" % origin,
                     id="4_0.E001",
                 )
             )
     return errors
```