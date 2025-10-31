# Bug Report: CORSMiddleware Whitespace Handling in allow_headers

**Target**: `starlette.middleware.cors.CORSMiddleware`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

CORSMiddleware incorrectly rejects CORS preflight requests when the `allow_headers` configuration contains headers with leading or trailing whitespace. The middleware preserves whitespace in the configured headers but strips whitespace when validating incoming requests, causing a mismatch.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from starlette.middleware.cors import CORSMiddleware
from starlette.datastructures import Headers


@given(st.text(alphabet=st.characters(min_codepoint=ord('a'), max_codepoint=ord('z')), min_size=1, max_size=20))
def test_cors_headers_whitespace_handling(header_name):
    header_with_space = header_name + "  "

    cors = CORSMiddleware(
        app=lambda s, r, sn: None,
        allow_headers=[header_with_space]
    )

    request_headers = Headers(raw=[
        (b"origin", b"http://example.com"),
        (b"access-control-request-method", b"GET"),
        (b"access-control-request-headers", header_name.encode())
    ])

    response = cors.preflight_response(request_headers)

    assert response.status_code == 200, (
        f"Header '{header_name}' should be allowed when configured as '{header_with_space}'. "
        f"Got status {response.status_code}"
    )
```

**Failing input**: `header_name='a'` (or any header name with whitespace added to configured allow_headers)

## Reproducing the Bug

```python
from starlette.middleware.cors import CORSMiddleware
from starlette.datastructures import Headers

cors = CORSMiddleware(
    app=lambda s, r, sn: None,
    allow_headers=["X-Custom-Header "]
)

request_headers = Headers(raw=[
    (b"origin", b"http://example.com"),
    (b"access-control-request-method", b"GET"),
    (b"access-control-request-headers", b"X-Custom-Header")
])

response = cors.preflight_response(request_headers)

print(f"Status: {response.status_code}")
print(f"Expected: 200, Got: {response.status_code}")
```

## Why This Is A Bug

The bug occurs because:

1. In `__init__` (line 67): `self.allow_headers = [h.lower() for h in allow_headers]` - headers are lowercased but NOT stripped
2. In `preflight_response` (lines 128-129): The validation code strips whitespace from incoming headers before checking

This creates an inconsistency:
- Configured: `["x-custom-header "]` (with trailing space)
- Incoming request header after processing: `"x-custom-header"` (stripped)
- Check: `"x-custom-header" not in ["x-custom-header "]` → True → REJECTED

While users shouldn't include whitespace in header names, the middleware should be defensive and normalize the configuration to prevent this inconsistency.

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