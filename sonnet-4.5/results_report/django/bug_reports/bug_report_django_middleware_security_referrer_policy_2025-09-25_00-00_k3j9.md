# Bug Report: Django SecurityMiddleware Referrer-Policy Inconsistent Whitespace Handling

**Target**: `django.middleware.security.SecurityMiddleware.process_response`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The SecurityMiddleware handles whitespace inconsistently when processing the `SECURE_REFERRER_POLICY` setting. When the setting is a string, whitespace is stripped from each comma-separated value. When the setting is a list or iterable, whitespace is preserved in each value. This leads to different HTTP header values for logically equivalent configurations.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'test_settings')

from unittest.mock import Mock

from hypothesis import given, settings, strategies as st

import django
from django.conf import settings as django_settings

if not django_settings.configured:
    django_settings.configure(
        DEBUG=True,
        SECRET_KEY='test-secret-key',
        SECURE_REFERRER_POLICY=None,
        SECURE_HSTS_SECONDS=0,
        SECURE_HSTS_INCLUDE_SUBDOMAINS=False,
        SECURE_HSTS_PRELOAD=False,
        SECURE_CONTENT_TYPE_NOSNIFF=False,
        SECURE_SSL_REDIRECT=False,
        SECURE_SSL_HOST=None,
        SECURE_REDIRECT_EXEMPT=[],
        SECURE_CROSS_ORIGIN_OPENER_POLICY=None,
    )
    django.setup()

from django.middleware.security import SecurityMiddleware
from django.http import HttpResponse, HttpRequest


@given(st.lists(
    st.text(alphabet=st.characters(min_codepoint=32, max_codepoint=126), min_size=1, max_size=50),
    min_size=1,
    max_size=5
))
@settings(max_examples=200)
def test_referrer_policy_whitespace_handling_consistency(policy_values):
    get_response = Mock()

    policy_str = ','.join(policy_values)
    policy_list = list(policy_values)

    django_settings.SECURE_REFERRER_POLICY = policy_str
    middleware_str = SecurityMiddleware(get_response)
    request_str = Mock(spec=HttpRequest)
    request_str.is_secure.return_value = False
    response_str = HttpResponse()
    result_str = middleware_str.process_response(request_str, response_str)
    header_from_str = result_str.get('Referrer-Policy')

    django_settings.SECURE_REFERRER_POLICY = policy_list
    middleware_list = SecurityMiddleware(get_response)
    request_list = Mock(spec=HttpRequest)
    request_list.is_secure.return_value = False
    response_list = HttpResponse()
    result_list = middleware_list.process_response(request_list, response_list)
    header_from_list = result_list.get('Referrer-Policy')

    assert header_from_str == header_from_list
```

**Failing input**: `policy_values=[' ']`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'test_settings')

from unittest.mock import Mock

import django
from django.conf import settings as django_settings

if not django_settings.configured:
    django_settings.configure(
        DEBUG=True,
        SECRET_KEY='test-secret-key',
        SECURE_REFERRER_POLICY=None,
        SECURE_HSTS_SECONDS=0,
        SECURE_HSTS_INCLUDE_SUBDOMAINS=False,
        SECURE_HSTS_PRELOAD=False,
        SECURE_CONTENT_TYPE_NOSNIFF=False,
        SECURE_SSL_REDIRECT=False,
        SECURE_SSL_HOST=None,
        SECURE_REDIRECT_EXEMPT=[],
        SECURE_CROSS_ORIGIN_OPENER_POLICY=None,
    )
    django.setup()

from django.middleware.security import SecurityMiddleware
from django.http import HttpResponse, HttpRequest

django_settings.SECURE_REFERRER_POLICY = [" no-referrer ", "  strict-origin"]
middleware = SecurityMiddleware(Mock())
request = Mock(spec=HttpRequest)
request.is_secure.return_value = False
response = HttpResponse()
result = middleware.process_response(request, response)

print(f"List input: {django_settings.SECURE_REFERRER_POLICY}")
print(f"Header value: '{result.get('Referrer-Policy')}'")
print(f"Result: ' no-referrer ,  strict-origin' (whitespace preserved)\n")

django_settings.SECURE_REFERRER_POLICY = " no-referrer ,  strict-origin"
middleware = SecurityMiddleware(Mock())
request = Mock(spec=HttpRequest)
request.is_secure.return_value = False
response = HttpResponse()
result = middleware.process_response(request, response)

print(f"String input: '{django_settings.SECURE_REFERRER_POLICY}'")
print(f"Header value: '{result.get('Referrer-Policy')}'")
print(f"Result: 'no-referrer,strict-origin' (whitespace stripped)")
```

## Why This Is A Bug

The Django documentation states that `SECURE_REFERRER_POLICY` can accept either a string or an iterable. Users would reasonably expect that `["no-referrer", "strict-origin"]` and `"no-referrer, strict-origin"` would produce identical HTTP headers. However, the implementation treats these differently when whitespace is present:

- String values: each comma-separated value is stripped
- List/iterable values: values are joined as-is without stripping

This inconsistency violates the principle of least surprise and could lead to invalid HTTP headers when users provide list values with accidental whitespace.

## Fix

```diff
--- a/django/middleware/security.py
+++ b/django/middleware/security.py
@@ -49,11 +49,12 @@ class SecurityMiddleware(MiddlewareMixin):
         if self.referrer_policy:
             # Support a comma-separated string or iterable of values to allow
             # fallback.
+            values = (
+                [v.strip() for v in self.referrer_policy.split(",")]
+                if isinstance(self.referrer_policy, str)
+                else [v.strip() for v in self.referrer_policy]
+            )
             response.headers.setdefault(
                 "Referrer-Policy",
-                ",".join(
-                    [v.strip() for v in self.referrer_policy.split(",")]
-                    if isinstance(self.referrer_policy, str)
-                    else self.referrer_policy
-                ),
+                ",".join(values),
             )
```