# Bug Report: CORSMiddleware Header Validation Inconsistency

**Target**: `starlette.middleware.cors.CORSMiddleware.preflight_response`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

CORSMiddleware has an inconsistency in how it processes allowed headers: headers are lowercased but not stripped when stored, but are lowercased and stripped when validated. This causes headers with leading/trailing whitespace to be rejected even when explicitly allowed.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st
from starlette.datastructures import Headers
from starlette.middleware.cors import CORSMiddleware

@given(st.lists(st.text(alphabet=st.characters(max_codepoint=255, blacklist_categories=('Cc', 'Cs')), min_size=1, max_size=20), min_size=0, max_size=5))
@settings(max_examples=200)
def test_cors_preflight_headers_validation(allowed_headers):
    middleware = CORSMiddleware(
        None,
        allow_origins=["https://example.com"],
        allow_methods=["GET"],
        allow_headers=allowed_headers
    )

    requested_headers = ", ".join(allowed_headers) if allowed_headers else ""

    headers_dict = {
        "origin": "https://example.com",
        "access-control-request-method": "GET"
    }

    if requested_headers:
        headers_dict["access-control-request-headers"] = requested_headers

    headers = Headers(headers_dict)
    response = middleware.preflight_response(headers)

    assert response.status_code == 200
```

**Failing input**: `allowed_headers=['\xa0']` (or any header with leading/trailing whitespace like `' X-Custom-Header '`)

## Reproducing the Bug

```python
from starlette.datastructures import Headers
from starlette.middleware.cors import CORSMiddleware

allowed_headers = [' X-Custom-Header ']

middleware = CORSMiddleware(
    None,
    allow_origins=["https://example.com"],
    allow_methods=["GET"],
    allow_headers=allowed_headers
)

headers = Headers({
    "origin": "https://example.com",
    "access-control-request-method": "GET",
    "access-control-request-headers": " X-Custom-Header "
})

response = middleware.preflight_response(headers)

print(response.status_code)

assert response.status_code == 200
```

## Why This Is A Bug

The code has an inconsistency in processing allowed headers:

1. In `__init__` (line 67): `self.allow_headers = [h.lower() for h in allow_headers]` - headers are lowercased but NOT stripped
2. In `preflight_response` (line 129): `if header.strip() not in self.allow_headers` - headers are lowercased AND stripped before checking

This creates a mismatch where a header like `' X-Custom-Header '` becomes `' x-custom-header '` when stored, but `'x-custom-header'` (no spaces) when checked, causing the validation to fail.

## Fix

```diff
--- a/starlette/middleware/cors.py
+++ b/starlette/middleware/cors.py
@@ -64,7 +64,7 @@ class CORSMiddleware:
         self.app = app
         self.allow_origins = allow_origins
         self.allow_methods = allow_methods
-        self.allow_headers = [h.lower() for h in allow_headers]
+        self.allow_headers = [h.lower().strip() for h in allow_headers]
         self.allow_all_origins = allow_all_origins
         self.allow_all_headers = allow_all_headers
         self.preflight_explicit_allow_origin = preflight_explicit_allow_origin
```