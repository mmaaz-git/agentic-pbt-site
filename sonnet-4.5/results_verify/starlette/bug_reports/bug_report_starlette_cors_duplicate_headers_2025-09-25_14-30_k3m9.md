# Bug Report: CORSMiddleware Duplicate Headers

**Target**: `starlette.middleware.cors.CORSMiddleware`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

CORSMiddleware can produce duplicate headers in its `allow_headers` attribute when users provide headers that differ only in case. Since HTTP headers are case-insensitive (per RFC 2616), headers like `'X-Custom-Header'` and `'x-custom-header'` should be treated as the same header, but the middleware creates duplicates.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st
from starlette.middleware.cors import CORSMiddleware, SAFELISTED_HEADERS


def dummy_app(scope, receive, send):
    pass


@given(st.lists(st.text(alphabet=st.characters(min_codepoint=1, blacklist_categories=('Cc', 'Cs')))))
@settings(max_examples=1000)
def test_cors_allow_headers_no_duplicates(headers):
    middleware = CORSMiddleware(dummy_app, allow_headers=headers)
    assert len(middleware.allow_headers) == len(set(middleware.allow_headers))
```

**Failing input**: `['F', 'f']` or `['X-Custom-Header', 'x-custom-header']`

## Reproducing the Bug

```python
from starlette.middleware.cors import CORSMiddleware


def dummy_app(scope, receive, send):
    pass


middleware = CORSMiddleware(dummy_app, allow_headers=['X-Custom-Header', 'x-custom-header'])
print(middleware.allow_headers)
```

Output:
```
['accept', 'accept-language', 'content-language', 'content-type', 'x-custom-header', 'x-custom-header']
```

The list contains duplicate `'x-custom-header'` entries.

## Why This Is A Bug

HTTP headers are case-insensitive per RFC 2616. When a user provides headers that differ only in case (e.g., `'X-Custom-Header'` and `'x-custom-header'`), they should be treated as a single header. However, the current implementation:

1. Performs set union on the original (case-preserved) headers: `set(['X-Custom-Header', 'x-custom-header'])` → `{'X-Custom-Header', 'x-custom-header'}`
2. Sorts them: `['X-Custom-Header', 'x-custom-header']`
3. Lowercases them: `['x-custom-header', 'x-custom-header']`

This creates duplicates in the final list, which violates the expectation that allowed headers should be unique.

## Fix

The bug occurs in the `__init__` method of `CORSMiddleware` at lines 44-53 of `cors.py`. The fix is to lowercase headers before performing the set union and sorting:

```diff
--- a/starlette/middleware/cors.py
+++ b/starlette/middleware/cors.py
@@ -41,11 +41,11 @@ class CORSMiddleware:
             preflight_headers["Access-Control-Allow-Origin"] = "*"
         preflight_headers.update(
             {
                 "Access-Control-Allow-Methods": ", ".join(allow_methods),
                 "Access-Control-Max-Age": str(max_age),
             }
         )
-        allow_headers = sorted(SAFELISTED_HEADERS | set(allow_headers))
+        allow_headers = sorted(SAFELISTED_HEADERS | {h.lower() for h in allow_headers})
         if allow_headers and not allow_all_headers:
             preflight_headers["Access-Control-Allow-Headers"] = ", ".join(allow_headers)
         if allow_credentials:
             preflight_headers["Access-Control-Allow-Credentials"] = "true"

         self.app = app
@@ -50,7 +50,7 @@ class CORSMiddleware:
         self.allow_all_headers = allow_all_headers
         self.preflight_explicit_allow_origin = preflight_explicit_allow_origin
         self.allow_origin_regex = compiled_allow_origin_regex
         self.simple_headers = simple_headers
         self.preflight_headers = preflight_headers
-        self.allow_headers = [h.lower() for h in allow_headers]
+        self.allow_headers = allow_headers
```

This change:
1. Lowercases headers before adding them to the set: `{h.lower() for h in allow_headers}`
2. Removes the redundant lowercasing step later since headers are already lowercase
3. Ensures that headers differing only in case are treated as duplicates and removed by the set operation