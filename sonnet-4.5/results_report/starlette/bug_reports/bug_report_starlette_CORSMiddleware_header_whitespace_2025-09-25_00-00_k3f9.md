# Bug Report: starlette.middleware.cors.CORSMiddleware Header Whitespace Inconsistency

**Target**: `starlette.middleware.cors.CORSMiddleware`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`CORSMiddleware` incorrectly rejects valid CORS preflight requests when `allow_headers` contains header names with leading/trailing whitespace. The middleware stores configured headers without stripping whitespace but strips whitespace from request headers during comparison, causing a mismatch.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from starlette.middleware.cors import CORSMiddleware
from starlette.datastructures import Headers


def dummy_app(scope, receive, send):
    pass


@given(st.text(min_size=1, max_size=20, alphabet=st.characters(min_codepoint=97, max_codepoint=122)))
def test_cors_header_whitespace_consistency(header_name):
    cors_with_spaces = CORSMiddleware(
        app=dummy_app,
        allow_origins=["https://example.com"],
        allow_headers=[f" {header_name} "],
    )

    cors_without_spaces = CORSMiddleware(
        app=dummy_app,
        allow_origins=["https://example.com"],
        allow_headers=[header_name],
    )

    headers_requesting = Headers(raw=[
        (b"origin", b"https://example.com"),
        (b"access-control-request-method", b"GET"),
        (b"access-control-request-headers", header_name.encode()),
    ])

    response_without_spaces = cors_without_spaces.preflight_response(request_headers=headers_requesting)
    assert response_without_spaces.status_code == 200

    response_with_spaces = cors_with_spaces.preflight_response(request_headers=headers_requesting)
    assert response_with_spaces.status_code == 200, "Headers with spaces in config should still match"
```

**Failing input**: Any header name with leading/trailing spaces, e.g., `" X-Custom-Header "`

## Reproducing the Bug

```python
from starlette.middleware.cors import CORSMiddleware
from starlette.datastructures import Headers


def dummy_app(scope, receive, send):
    pass


cors = CORSMiddleware(
    app=dummy_app,
    allow_origins=["https://example.com"],
    allow_headers=[" X-Custom-Header "],
)

headers = Headers(raw=[
    (b"origin", b"https://example.com"),
    (b"access-control-request-method", b"GET"),
    (b"access-control-request-headers", b"X-Custom-Header"),
])

response = cors.preflight_response(request_headers=headers)

print(f"Status: {response.status_code}")
print(f"Body: {response.body}")
```

Output:
```
Status: 400
Body: b'Disallowed CORS headers'
```

Expected: `200 OK` because the header `X-Custom-Header` matches the configured ` X-Custom-Header ` (after normalization).

## Why This Is A Bug

The bug violates the expected property that whitespace in configuration should not affect functional behavior. Users may inadvertently add whitespace in configuration (e.g., from formatted lists or config files), and this should not break CORS functionality.

The root cause is in `starlette/middleware/cors.py`:

1. **Line 67**: Stores headers without stripping: `self.allow_headers = [h.lower() for h in allow_headers]`
2. **Line 128**: Strips request headers before comparison: `for header in [h.lower() for h in requested_headers.split(",")]:`
3. **Line 129**: Strips again: `if header.strip() not in self.allow_headers:`

This creates an inconsistency where `"x-custom-header".strip()` (from request) doesn't match `" x-custom-header "` (from config).

## Fix

```diff
--- a/starlette/middleware/cors.py
+++ b/starlette/middleware/cors.py
@@ -64,7 +64,7 @@ class CORSMiddleware:
         self.app = app
         self.allow_origins = allow_origins
         self.allow_methods = allow_methods
-        self.allow_headers = [h.lower() for h in allow_headers]
+        self.allow_headers = [h.strip().lower() for h in allow_headers]
         self.allow_all_origins = allow_all_origins
         self.allow_all_headers = allow_all_headers
         self.preflight_explicit_allow_origin = preflight_explicit_allow_origin
```

This ensures both configuration and request headers are normalized consistently by stripping whitespace.