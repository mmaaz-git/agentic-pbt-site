# Bug Report: CORSMiddleware Header Case Sensitivity

**Target**: `starlette.middleware.cors.CORSMiddleware` (exposed via `fastapi.middleware.cors.CORSMiddleware`)
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `CORSMiddleware` class produces inconsistent header ordering and casing in CORS responses depending on the case of input headers. This violates the principle that HTTP headers should be treated case-insensitively.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from starlette.middleware.cors import CORSMiddleware


class DummyApp:
    async def __call__(self, scope, receive, send):
        pass


@given(
    st.text(min_size=1, max_size=30, alphabet=st.characters(min_codepoint=65, max_codepoint=90))
)
@settings(max_examples=500)
def test_cors_allow_headers_case_variants(header_upper):
    app = DummyApp()

    cors_upper = CORSMiddleware(app, allow_headers=[header_upper])
    cors_lower = CORSMiddleware(app, allow_headers=[header_upper.lower()])
    cors_mixed = CORSMiddleware(app, allow_headers=[header_upper.title()])

    assert cors_upper.allow_headers == cors_lower.allow_headers == cors_mixed.allow_headers
```

**Failing input**: `header_upper='A'`

## Reproducing the Bug

```python
from starlette.middleware.cors import CORSMiddleware


class DummyApp:
    async def __call__(self, scope, receive, send):
        pass


app = DummyApp()

cors_upper = CORSMiddleware(app, allow_headers=['A'])
cors_lower = CORSMiddleware(app, allow_headers=['a'])

print("With uppercase 'A':")
print(f"  allow_headers: {cors_upper.allow_headers}")
print(f"  preflight: {cors_upper.preflight_headers['Access-Control-Allow-Headers']}")

print("\nWith lowercase 'a':")
print(f"  allow_headers: {cors_lower.allow_headers}")
print(f"  preflight: {cors_lower.preflight_headers['Access-Control-Allow-Headers']}")

print("\nBUG - Different order and case:")
print(f"  allow_headers equal? {cors_upper.allow_headers == cors_lower.allow_headers}")
print(f"  preflight equal? {cors_upper.preflight_headers['Access-Control-Allow-Headers'] == cors_lower.preflight_headers['Access-Control-Allow-Headers']}")
```

Output:
```
With uppercase 'A':
  allow_headers: ['a', 'accept', 'accept-language', 'content-language', 'content-type']
  preflight: A, Accept, Accept-Language, Content-Language, Content-Type

With lowercase 'a':
  allow_headers: ['accept', 'accept-language', 'content-language', 'content-type', 'a']
  preflight: Accept, Accept-Language, Content-Language, Content-Type, a

BUG - Different order and case:
  allow_headers equal? False
  preflight equal? False
```

## Why This Is A Bug

HTTP headers are case-insensitive per RFC 7230. The middleware should treat `allow_headers=['A']` and `allow_headers=['a']` identically, producing the same internal state and response headers. However, the current implementation sorts headers before lowercasing them, causing:

1. Different ordering in `self.allow_headers` based on input case
2. Different ordering and casing in CORS preflight responses
3. Unpredictable behavior for API consumers

The root cause is in lines 74-82 of `cors.py`:

```python
allow_headers = sorted(SAFELISTED_HEADERS | set(allow_headers))  # Sort before lowercase
if allow_headers and not allow_all_headers:
    preflight_headers["Access-Control-Allow-Headers"] = ", ".join(allow_headers)
# ...
self.allow_headers = [h.lower() for h in allow_headers]  # Lowercase after sort
```

Since Python's `sorted()` is case-sensitive (uppercase < lowercase in ASCII), headers are sorted differently based on their case.

## Fix

```diff
--- a/starlette/middleware/cors.py
+++ b/starlette/middleware/cors.py
@@ -71,7 +71,9 @@ class CORSMiddleware:
                 "Access-Control-Max-Age": str(max_age),
             }
         )
-        allow_headers = sorted(SAFELISTED_HEADERS | set(allow_headers))
+        # Lowercase headers first for case-insensitive sorting
+        lowercased_safelisted = {h.lower() for h in SAFELISTED_HEADERS}
+        allow_headers = sorted(lowercased_safelisted | {h.lower() for h in allow_headers})
         if allow_headers and not allow_all_headers:
             preflight_headers["Access-Control-Allow-Headers"] = ", ".join(allow_headers)
         if allow_credentials:
@@ -79,7 +81,7 @@ class CORSMiddleware:
         self.app = app
         self.allow_origins = allow_origins
         self.allow_methods = allow_methods
-        self.allow_headers = [h.lower() for h in allow_headers]
+        self.allow_headers = allow_headers
         self.allow_all_origins = allow_all_origins
         self.allow_all_headers = allow_all_headers
         self.preflight_explicit_allow_origin = preflight_explicit_allow_origin
```

This ensures headers are lowercased before sorting, producing consistent behavior regardless of input case.