# Bug Report: starlette.middleware.cors.CORSMiddleware Header Whitespace Inconsistency

**Target**: `starlette.middleware.cors.CORSMiddleware`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

CORSMiddleware incorrectly rejects valid CORS preflight requests when `allow_headers` configuration contains header names with leading/trailing whitespace, due to asymmetric whitespace handling between configuration and request headers.

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


if __name__ == "__main__":
    test_cors_header_whitespace_consistency()
```

<details>

<summary>
**Failing input**: `'a'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/32/hypo.py", line 38, in <module>
    test_cors_header_whitespace_consistency()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/32/hypo.py", line 11, in test_cors_header_whitespace_consistency
    def test_cors_header_whitespace_consistency(header_name):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/32/hypo.py", line 34, in test_cors_header_whitespace_consistency
    assert response_with_spaces.status_code == 200, "Headers with spaces in config should still match"
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Headers with spaces in config should still match
Falsifying example: test_cors_header_whitespace_consistency(
    header_name='a',  # or any other generated value
)
```
</details>

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

<details>

<summary>
CORS preflight request rejected with 400 error
</summary>
```
Status: 400
Body: b'Disallowed CORS headers'
```
</details>

## Why This Is A Bug

This violates expected behavior because the middleware exhibits inconsistent whitespace handling that breaks legitimate CORS requests. The code strips whitespace from incoming request headers but does not strip whitespace from configuration headers, creating an impossible-to-match condition when configuration contains whitespace.

Specifically:
- **Line 67** (`starlette/middleware/cors.py`): Configuration headers are stored with only lowercasing: `self.allow_headers = [h.lower() for h in allow_headers]`
- **Line 129**: Request headers are stripped before comparison: `if header.strip() not in self.allow_headers:`

This means a configuration header like `" x-custom-header "` (with spaces) will never match a request header `"x-custom-header"` because the request header becomes `"x-custom-header"` after stripping but is compared against `" x-custom-header "` in the allow list.

The HTTP/1.1 specification (RFC 7230, Section 3.2) states that whitespace surrounding header field values should be excluded when parsing, making this behavior inconsistent with HTTP standards. Users may inadvertently include whitespace when loading configuration from formatted files, YAML configurations, or multi-line strings, causing valid CORS requests to be incorrectly rejected.

## Relevant Context

The bug occurs in the CORS (Cross-Origin Resource Sharing) middleware, which is critical for web security. CORS controls which origins can access resources from a different domain, and incorrect CORS handling can either:
1. Block legitimate requests (as in this bug) - causing functionality issues
2. Allow unauthorized requests - causing security issues

The middleware already demonstrates intent to normalize headers:
- It lowercases all headers for case-insensitive comparison (line 67)
- It strips whitespace from request headers (line 129)
- It handles the special case of wildcard headers (line 69)

The missing strip operation on configuration headers appears to be an oversight rather than intentional design. The SAFELISTED_HEADERS constant (line 12) contains properly formatted header names without whitespace, suggesting the expected format.

Documentation: https://www.starlette.io/middleware/#corsmiddleware
Source code: https://github.com/encode/starlette/blob/master/starlette/middleware/cors.py

## Proposed Fix

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