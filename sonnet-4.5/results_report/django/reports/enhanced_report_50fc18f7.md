# Bug Report: Django CSRF Trusted Origins Incomplete Scheme Validation

**Target**: `django.core.checks.compatibility.django_4_0.check_csrf_trusted_origins`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `check_csrf_trusted_origins` function incorrectly validates CSRF_TRUSTED_ORIGINS entries by only checking if `"://"` appears anywhere in the string, rather than verifying that a non-empty scheme exists at the beginning of the origin URL.

## Property-Based Test

```python
#!/usr/bin/env python3
"""
Hypothesis test for Django CSRF trusted origins validation bug.
This test verifies that origins must have a scheme at the beginning.
"""

import os
os.environ['DJANGO_SETTINGS_MODULE'] = 'test_settings'

from django.conf import settings
if not settings.configured:
    settings.configure(
        SECRET_KEY='test-secret-key',
        CSRF_TRUSTED_ORIGINS=[],
        SILENCED_SYSTEM_CHECKS=[],
    )

import django
django.setup()

from hypothesis import given, strategies as st, example
from unittest.mock import patch
from django.core.checks.compatibility.django_4_0 import check_csrf_trusted_origins

@given(st.text())
@example("://example.com")  # Explicit failing case
def test_scheme_must_be_at_start(origin):
    """
    Test that CSRF_TRUSTED_ORIGINS validation properly checks for scheme at start.

    The Django documentation states that origins must start with a scheme,
    and the error message says "must start with a scheme", but the actual
    validation only checks if "://" exists anywhere in the string.
    """
    with patch('django.conf.settings.CSRF_TRUSTED_ORIGINS', [origin]):
        errors = check_csrf_trusted_origins(app_configs=None)

        if '://' in origin:
            index_of_separator = origin.index('://')
            if index_of_separator == 0:
                # If :// is at the start, there's no scheme before it
                assert len(errors) > 0, \
                    f"Origin '{origin}' has no scheme before ://, should fail"
        else:
            # If there's no :// at all, it should fail
            assert len(errors) > 0, \
                f"Origin '{origin}' has no ://, should fail"

if __name__ == "__main__":
    # Run the test with hypothesis
    test_scheme_must_be_at_start()
```

<details>

<summary>
**Failing input**: `'://example.com'`
</summary>
```
============================= test session starts ==============================
platform linux -- Python 3.13.2, pytest-8.4.1, pluggy-1.5.0 -- /home/npc/miniconda/bin/python3
cachedir: .pytest_cache
hypothesis profile 'default'
rootdir: /home/npc/pbt/agentic-pbt/worker_/44
plugins: anyio-4.9.0, hypothesis-6.139.1, asyncio-1.2.0, langsmith-0.4.29
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 1 item

hypo.py::test_scheme_must_be_at_start FAILED                             [100%]

=================================== FAILURES ===================================
_________________________ test_scheme_must_be_at_start _________________________
hypo.py:26: in test_scheme_must_be_at_start
    @example("://example.com")  # Explicit failing case
                   ^^^
/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py:1613: in _raise_to_user
    raise the_error_hypothesis_found
hypo.py:42: in test_scheme_must_be_at_start
    assert len(errors) > 0, \
E   AssertionError: Origin '://example.com' has no scheme before ://, should fail
E   assert 0 > 0
E    +  where 0 = len([])
E   Falsifying explicit example: test_scheme_must_be_at_start(
E       origin='://example.com',
E   )
=========================== short test summary info ============================
FAILED hypo.py::test_scheme_must_be_at_start - AssertionError: Origin '://exa...
============================== 1 failed in 0.19s ===============================
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""
Minimal reproduction of Django CSRF trusted origins validation bug.
This demonstrates that origins without a proper scheme are incorrectly accepted.
"""

import os
os.environ['DJANGO_SETTINGS_MODULE'] = 'test_settings'

from django.conf import settings
if not settings.configured:
    settings.configure(
        SECRET_KEY='test-secret-key',
        CSRF_TRUSTED_ORIGINS=[],
        SILENCED_SYSTEM_CHECKS=[],
    )

import django
django.setup()

from unittest.mock import patch
from django.core.checks.compatibility.django_4_0 import check_csrf_trusted_origins

# Test cases that should all produce errors
test_cases = [
    '://example.com',          # No scheme before ://
    'example.com://foo',       # Scheme not at start
    'example://com',           # :// in middle but not a valid URL format
    '://',                     # Just the separator
]

# Test case that should NOT produce an error
valid_cases = [
    'https://example.com',     # Proper scheme at start
    'http://localhost:8000',   # Proper scheme with port
]

print("=" * 60)
print("Django CSRF Trusted Origins Validation Bug Demonstration")
print("=" * 60)
print()

print("Testing INVALID origins that SHOULD produce errors:")
print("-" * 50)
for origin in test_cases:
    with patch('django.conf.settings.CSRF_TRUSTED_ORIGINS', [origin]):
        errors = check_csrf_trusted_origins(app_configs=None)
        if len(errors) == 0:
            print(f"❌ BUG: '{origin}' - NO ERROR (should fail validation)")
        else:
            print(f"✓ OK: '{origin}' - ERROR generated as expected")
print()

print("Testing VALID origins that should NOT produce errors:")
print("-" * 50)
for origin in valid_cases:
    with patch('django.conf.settings.CSRF_TRUSTED_ORIGINS', [origin]):
        errors = check_csrf_trusted_origins(app_configs=None)
        if len(errors) == 0:
            print(f"✓ OK: '{origin}' - No error as expected")
        else:
            print(f"❌ UNEXPECTED: '{origin}' - ERROR generated")
            print(f"   Error: {errors[0].msg}")

print()
print("=" * 60)
print("CONCLUSION: The validation accepts malformed origins with '://'")
print("anywhere in the string, even without a proper scheme at the start.")
print("=" * 60)
```

<details>

<summary>
Validation incorrectly accepts malformed origins
</summary>
```
============================================================
Django CSRF Trusted Origins Validation Bug Demonstration
============================================================

Testing INVALID origins that SHOULD produce errors:
--------------------------------------------------
❌ BUG: '://example.com' - NO ERROR (should fail validation)
❌ BUG: 'example.com://foo' - NO ERROR (should fail validation)
❌ BUG: 'example://com' - NO ERROR (should fail validation)
❌ BUG: '://' - NO ERROR (should fail validation)

Testing VALID origins that should NOT produce errors:
--------------------------------------------------
✓ OK: 'https://example.com' - No error as expected
✓ OK: 'http://localhost:8000' - No error as expected

============================================================
CONCLUSION: The validation accepts malformed origins with '://'
anywhere in the string, even without a proper scheme at the start.
============================================================
```
</details>

## Why This Is A Bug

The validation function at line 10 of `/django/core/checks/compatibility/django_4_0.py` only checks `if "://" not in origin`, which fails to enforce the documented requirement that origins "must start with a scheme". This contradicts:

1. **The error message itself** (lines 13-15) which explicitly states: "As of Django 4.0, the values in the CSRF_TRUSTED_ORIGINS setting **must start with a scheme** (usually http:// or https://)"

2. **Django's official documentation** which requires origins to include "the full URL scheme and domain" with examples like `'https://subdomain.example.com'`

3. **The intended security behavior** where Django's CSRF middleware uses `urlsplit()` to parse these origins. When given malformed origins:
   - `'://example.com'` is parsed with empty scheme and netloc, treating the whole string as a path
   - `'example.com://foo'` is parsed with 'example.com' as the scheme and 'foo' as the netloc
   - This causes the CSRF protection to potentially fail or behave unexpectedly

The validation accepts any string containing `"://"` regardless of position or whether a valid scheme precedes it. This allows developers to unknowingly misconfigure CSRF protection without receiving the intended warning.

## Relevant Context

The bug exists in Django's compatibility check system introduced in Django 4.0 to help developers migrate from the previous format (domain-only) to the new format (scheme + domain). The check was meant to catch configurations like `'example.com'` and suggest `'https://example.com'`, but its implementation is too permissive.

Django's CSRF middleware relies on properly formatted origins to correctly identify trusted sources for unsafe HTTP methods (POST, PUT, DELETE, etc.). Malformed origins could lead to:
- CSRF protection being bypassed if the malformed origin matches incoming requests unexpectedly
- CSRF protection being too restrictive if the malformed origin never matches legitimate requests
- Silent failures where developers believe their configuration is correct but it's not working as intended

Relevant Django source code location: `/django/core/checks/compatibility/django_4_0.py`

## Proposed Fix

```diff
--- a/django/core/checks/compatibility/django_4_0.py
+++ b/django/core/checks/compatibility/django_4_0.py
@@ -7,7 +7,10 @@ from .. import Error, Tags, register
 def check_csrf_trusted_origins(app_configs, **kwargs):
     errors = []
     for origin in settings.CSRF_TRUSTED_ORIGINS:
-        if "://" not in origin:
+        # Check if :// exists and has a non-empty scheme before it
+        separator_index = origin.find("://")
+        # Origin must contain :// and have at least one character before it
+        if separator_index <= 0:
             errors.append(
                 Error(
                     "As of Django 4.0, the values in the CSRF_TRUSTED_ORIGINS "
```