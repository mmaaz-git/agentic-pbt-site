# Bug Report: BrokenLinkEmailsMiddleware Incorrect Referer Comparison

**Target**: `django.middleware.common.BrokenLinkEmailsMiddleware.is_ignorable_request`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `BrokenLinkEmailsMiddleware.is_ignorable_request` method incorrectly compares a full URL (referer) with a path-only string (uri), causing internal redirects from `APPEND_SLASH` to incorrectly trigger broken link notification emails.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from django.middleware.common import BrokenLinkEmailsMiddleware


@given(
    st.text(alphabet="abcdefghijklmnopqrstuvwxyz/", min_size=2, max_size=50).filter(lambda s: '/' in s),
    st.sampled_from(["http", "https"]),
    st.text(alphabet="abcdefghijklmnopqrstuvwxyz.-", min_size=4, max_size=30).filter(lambda s: '.' in s)
)
def test_is_ignorable_request_append_slash_internal_redirect(path, scheme, domain):
    if not path.startswith('/'):
        path = '/' + path
    if not path.endswith('/'):
        path = path + '/'

    middleware = BrokenLinkEmailsMiddleware(lambda r: r)

    class MockRequest:
        pass

    request = MockRequest()
    uri = path
    referer_full_url = f"{scheme}://{domain}{path[:-1]}"

    result = middleware.is_ignorable_request(request, uri, domain, referer_full_url)

    assert result == True, (
        f"Internal redirect from APPEND_SLASH should be ignorable, but got {result}. "
        f"URI: {uri}, Referer: {referer_full_url}"
    )
```

**Failing input**: All inputs fail. For example: `uri="/page/"`, `referer="http://example.com/page"`, `domain="example.com"`

## Reproducing the Bug

```python
from django.middleware.common import BrokenLinkEmailsMiddleware
from django.conf import settings

settings.configure(
    DEBUG=False,
    APPEND_SLASH=True,
    IGNORABLE_404_URLS=[],
)

middleware = BrokenLinkEmailsMiddleware(lambda r: r)

class MockRequest:
    pass

request = MockRequest()
domain = "example.com"
uri = "/page/"
referer = "http://example.com/page"

result = middleware.is_ignorable_request(request, uri, domain, referer)

print(f"URI: {uri}")
print(f"Referer: {referer}")
print(f"Should be ignorable: True")
print(f"Actually ignorable: {result}")
```

## Why This Is A Bug

Line 165 in `django/middleware/common.py`:

```python
if settings.APPEND_SLASH and uri.endswith("/") and referer == uri[:-1]:
    return True
```

This compares `referer` (a full URL like `"http://example.com/page"`) with `uri[:-1]` (a path like `"/page"`). These will never match because one includes the scheme and domain while the other doesn't.

The intended behavior is to identify internal redirects caused by `APPEND_SLASH` and mark them as ignorable (don't send broken link emails). However, this comparison fails 100% of the time with real HTTP Referer headers, causing Django to incorrectly send broken link emails for normal APPEND_SLASH redirects.

## Fix

```diff
--- a/django/middleware/common.py
+++ b/django/middleware/common.py
@@ -162,7 +162,12 @@ class BrokenLinkEmailsMiddleware(MiddlewareMixin):

         # APPEND_SLASH is enabled and the referer is equal to the current URL
         # without a trailing slash indicating an internal redirect.
-        if settings.APPEND_SLASH and uri.endswith("/") and referer == uri[:-1]:
+        if settings.APPEND_SLASH and uri.endswith("/"):
+            # Parse the referer to extract just the path for comparison
+            parsed_referer = urlsplit(referer)
+            if (parsed_referer.netloc in ["", domain] and
+                parsed_referer.path == uri[:-1]):
+                return True
+        if False:  # Remove old broken check
             return True

         # A '?' in referer is identified as a search engine source.
```

Or more concisely:

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