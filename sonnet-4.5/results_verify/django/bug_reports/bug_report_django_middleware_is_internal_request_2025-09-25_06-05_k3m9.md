# Bug Report: django.middleware.common BrokenLinkEmailsMiddleware is_internal_request Incorrectly Classifies URLs Without Trailing Slash

**Target**: `django.middleware.common.BrokenLinkEmailsMiddleware.is_internal_request`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `is_internal_request()` method incorrectly classifies same-domain URLs without a trailing slash as external requests, causing internal broken link notifications to omit the "INTERNAL" label in emails to site managers.

## Property-Based Test

```python
from hypothesis import given, strategies as st, example, settings
from django.middleware.common import BrokenLinkEmailsMiddleware

middleware = BrokenLinkEmailsMiddleware(lambda r: None)
domain_strategy = st.from_regex(r'^[a-z0-9][a-z0-9\-]*(\.[a-z0-9][a-z0-9\-]*)*$', fullmatch=True)

@settings(max_examples=200)
@given(domain=domain_strategy, scheme=st.sampled_from(['http', 'https']))
@example(domain='example.com', scheme='https')
def test_is_internal_request_without_trailing_slash(domain, scheme):
    referer_with_slash = f"{scheme}://{domain}/"
    referer_without_slash = f"{scheme}://{domain}"

    result_with_slash = middleware.is_internal_request(domain, referer_with_slash)
    result_without_slash = middleware.is_internal_request(domain, referer_without_slash)

    assert result_with_slash, f"{referer_with_slash} should be internal"
    assert result_without_slash, (
        f"{referer_without_slash} should be internal but is_internal_request() "
        f"returned False"
    )
```

**Failing input**: `domain='example.com', scheme='https'`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

from django.middleware.common import BrokenLinkEmailsMiddleware

middleware = BrokenLinkEmailsMiddleware(lambda r: None)
domain = "example.com"

referer_with_slash = "https://example.com/"
referer_without_slash = "https://example.com"

result_with = middleware.is_internal_request(domain, referer_with_slash)
result_without = middleware.is_internal_request(domain, referer_without_slash)

print(f"With slash:    {referer_with_slash} → {result_with}")
print(f"Without slash: {referer_without_slash} → {result_without}")

assert result_with == True
assert result_without == False
```

## Why This Is A Bug

The method's docstring states "Return True if the referring URL is the same domain as the current request." URLs with and without trailing slashes represent the same domain according to RFC 3986. `https://example.com` and `https://example.com/` are equivalent URLs pointing to the same resource.

The regex pattern `^https?://%s/` on line 151 requires a trailing slash, causing same-domain URLs without one to be incorrectly classified as external. This affects broken link email notifications sent to site managers, where internal broken links won't be marked with "INTERNAL" if the referer lacks a trailing slash.

## Fix

```diff
--- a/django/middleware/common.py
+++ b/django/middleware/common.py
@@ -148,7 +148,7 @@ class BrokenLinkEmailsMiddleware(MiddlewareMixin):
         request.
         """
         # Different subdomains are treated as different domains.
-        return bool(re.match("^https?://%s/" % re.escape(domain), referer))
+        return bool(re.match("^https?://%s(/|$)" % re.escape(domain), referer))
```

The fix changes the regex pattern from `^https?://%s/` to `^https?://%s(/|$)`, which matches URLs with a trailing slash OR the end of the string, correctly identifying both `https://example.com` and `https://example.com/` as internal requests.