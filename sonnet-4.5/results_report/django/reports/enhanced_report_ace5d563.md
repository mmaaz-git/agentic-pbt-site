# Bug Report: django.middleware.common.BrokenLinkEmailsMiddleware String Comparison Type Mismatch in APPEND_SLASH Detection

**Target**: `django.middleware.common.BrokenLinkEmailsMiddleware.is_ignorable_request`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `is_ignorable_request` method incorrectly compares a full URL string (HTTP Referer header) with a path-only string when attempting to detect APPEND_SLASH internal redirects, causing the comparison to always fail and resulting in unnecessary broken link notification emails to site administrators.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from django.middleware.common import BrokenLinkEmailsMiddleware
from django.conf import settings

# Configure Django settings
settings.configure(
    DEBUG=False,
    APPEND_SLASH=True,
    IGNORABLE_404_URLS=[],
)

@given(
    st.text(alphabet="abcdefghijklmnopqrstuvwxyz/", min_size=2, max_size=50).filter(lambda s: '/' in s),
    st.sampled_from(["http", "https"]),
    st.text(alphabet="abcdefghijklmnopqrstuvwxyz.-", min_size=4, max_size=30).filter(lambda s: '.' in s)
)
def test_is_ignorable_request_append_slash_internal_redirect(path, scheme, domain):
    # Ensure path starts with / and ends with /
    if not path.startswith('/'):
        path = '/' + path
    if not path.endswith('/'):
        path = path + '/'

    # Create middleware instance
    middleware = BrokenLinkEmailsMiddleware(lambda r: r)

    # Mock request object
    class MockRequest:
        pass

    request = MockRequest()

    # Set up test parameters
    uri = path
    referer_full_url = f"{scheme}://{domain}{path[:-1]}"

    # Call the method
    result = middleware.is_ignorable_request(request, uri, domain, referer_full_url)

    # Assert the expected behavior
    assert result == True, (
        f"Internal redirect from APPEND_SLASH should be ignorable, but got {result}. "
        f"URI: {uri}, Referer: {referer_full_url}"
    )

# Run the test
if __name__ == "__main__":
    test_is_ignorable_request_append_slash_internal_redirect()
```

<details>

<summary>
**Failing input**: `path='a/', scheme='http', domain='aaa.'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/41/hypo.py", line 48, in <module>
    test_is_ignorable_request_append_slash_internal_redirect()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/41/hypo.py", line 13, in test_is_ignorable_request_append_slash_internal_redirect
    st.text(alphabet="abcdefghijklmnopqrstuvwxyz/", min_size=2, max_size=50).filter(lambda s: '/' in s),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/41/hypo.py", line 41, in test_is_ignorable_request_append_slash_internal_redirect
    assert result == True, (
           ^^^^^^^^^^^^^^
AssertionError: Internal redirect from APPEND_SLASH should be ignorable, but got False. URI: /a/, Referer: http://aaa./a
Falsifying example: test_is_ignorable_request_append_slash_internal_redirect(
    path='a/',  # or any other generated value
    scheme='http',  # or any other generated value
    domain='aaa.',  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
from django.middleware.common import BrokenLinkEmailsMiddleware
from django.conf import settings

# Configure Django settings
settings.configure(
    DEBUG=False,
    APPEND_SLASH=True,
    IGNORABLE_404_URLS=[],
)

# Create the middleware instance
middleware = BrokenLinkEmailsMiddleware(lambda r: r)

# Mock request object
class MockRequest:
    pass

request = MockRequest()

# Test case parameters
domain = "example.com"
uri = "/page/"
referer = "http://example.com/page"

# Call the method that has the bug
result = middleware.is_ignorable_request(request, uri, domain, referer)

# Output the results
print(f"Testing BrokenLinkEmailsMiddleware.is_ignorable_request")
print(f"=" * 60)
print(f"Input parameters:")
print(f"  domain: {domain}")
print(f"  uri: {uri}")
print(f"  referer: {referer}")
print(f"")
print(f"Expected behavior:")
print(f"  This should be an internal APPEND_SLASH redirect")
print(f"  The referer 'http://example.com/page' redirected to '/page/'")
print(f"  This should be ignorable (return True)")
print(f"")
print(f"Actual result: {result}")
print(f"Expected result: True")
print(f"")
print(f"Bug explanation:")
print(f"  The code compares referer == uri[:-1]")
print(f"  Which is: '{referer}' == '{uri[:-1]}'")
print(f"  A full URL can never equal a path-only string")
```

<details>

<summary>
BrokenLinkEmailsMiddleware returns False instead of True for APPEND_SLASH redirects
</summary>
```
Testing BrokenLinkEmailsMiddleware.is_ignorable_request
============================================================
Input parameters:
  domain: example.com
  uri: /page/
  referer: http://example.com/page

Expected behavior:
  This should be an internal APPEND_SLASH redirect
  The referer 'http://example.com/page' redirected to '/page/'
  This should be ignorable (return True)

Actual result: False
Expected result: True

Bug explanation:
  The code compares referer == uri[:-1]
  Which is: 'http://example.com/page' == '/page'
  A full URL can never equal a path-only string
```
</details>

## Why This Is A Bug

The bug violates Django's intended behavior documented in the inline comment at lines 163-164 of `django/middleware/common.py`: "APPEND_SLASH is enabled and the referer is equal to the current URL without a trailing slash indicating an internal redirect."

The problematic code at line 165 attempts to detect when Django's `APPEND_SLASH` feature has redirected from `/page` to `/page/`, which should be marked as ignorable to prevent unnecessary broken link emails. However, the comparison `referer == uri[:-1]` compares incompatible string types:

1. The `referer` variable contains the full HTTP Referer header value (e.g., `"http://example.com/page"`)
2. The `uri[:-1]` expression produces just the path without trailing slash (e.g., `"/page"`)

Since the HTTP Referer header is always a full URL per RFC 7231 Section 5.5.2, and the middleware passes `request.META.get("HTTP_REFERER", "")` directly to this method (line 124), this comparison will always fail in production environments.

This causes Django to send broken link notification emails to site managers for every legitimate APPEND_SLASH redirect, creating unnecessary noise and making the broken link monitoring feature less useful.

## Relevant Context

The BrokenLinkEmailsMiddleware is designed to help site administrators monitor genuine broken links on their website by sending email notifications for 404 errors. The APPEND_SLASH setting (enabled by default in Django) automatically redirects URLs without trailing slashes to their slash-appended versions when appropriate.

When both features are enabled, internal redirects from `/page` to `/page/` should not trigger broken link emails since they're intentional redirects, not actual broken links. The code attempts to implement this logic but fails due to the string comparison bug.

The bug affects production Django sites where:
- `DEBUG = False` (production mode)
- `APPEND_SLASH = True` (Django default)
- `BrokenLinkEmailsMiddleware` is enabled in `MIDDLEWARE`

Interestingly, the same file correctly handles URL parsing elsewhere. Line 174 uses `urlsplit(referer)` to properly parse the referer URL for a different check, demonstrating that the correct approach is already known and used in the same method.

## Proposed Fix

```diff
--- a/django/middleware/common.py
+++ b/django/middleware/common.py
@@ -162,7 +162,10 @@ class BrokenLinkEmailsMiddleware(MiddlewareMixin):

         # APPEND_SLASH is enabled and the referer is equal to the current URL
         # without a trailing slash indicating an internal redirect.
-        if settings.APPEND_SLASH and uri.endswith("/") and referer == uri[:-1]:
+        parsed_referer = urlsplit(referer)
+        if (settings.APPEND_SLASH and uri.endswith("/") and
+            parsed_referer.netloc in ["", domain] and
+            parsed_referer.path == uri[:-1]):
             return True

         # A '?' in referer is identified as a search engine source.
```