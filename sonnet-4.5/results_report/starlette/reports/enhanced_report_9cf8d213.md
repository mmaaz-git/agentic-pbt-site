# Bug Report: starlette.middleware.cors.CORSMiddleware Header Whitespace Processing Inconsistency

**Target**: `starlette.middleware.cors.CORSMiddleware.preflight_response`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

CORSMiddleware processes header names inconsistently: headers are lowercased but not stripped during initialization, yet they are both lowercased and stripped during validation, causing headers with whitespace to always fail validation even when explicitly allowed.

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

# Run the test
test_cors_preflight_headers_validation()
```

<details>

<summary>
**Failing input**: `allowed_headers=[',']`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/48/hypo.py", line 31, in <module>
    test_cors_preflight_headers_validation()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/48/hypo.py", line 6, in test_cors_preflight_headers_validation
    @settings(max_examples=200)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/48/hypo.py", line 28, in test_cors_preflight_headers_validation
    assert response.status_code == 200
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError
Falsifying example: test_cors_preflight_headers_validation(
    allowed_headers=[','],
)
```
</details>

## Reproducing the Bug

```python
from starlette.datastructures import Headers
from starlette.middleware.cors import CORSMiddleware

# Create middleware with a header that has leading/trailing spaces
allowed_headers = [' X-Custom-Header ']

middleware = CORSMiddleware(
    None,
    allow_origins=["https://example.com"],
    allow_methods=["GET"],
    allow_headers=allowed_headers
)

# Create headers for a preflight request with the same header (with spaces)
headers = Headers({
    "origin": "https://example.com",
    "access-control-request-method": "GET",
    "access-control-request-headers": " X-Custom-Header "
})

# Make the preflight request
response = middleware.preflight_response(headers)

print(f"Status code: {response.status_code}")
print(f"Response body: {response.body.decode()}")

# This should be 200 since we're requesting exactly what we allowed
# But it returns 400 because of the inconsistency
assert response.status_code == 200, f"Expected 200, got {response.status_code}"
```

<details>

<summary>
AssertionError: Expected 200, got 400
</summary>
```
Status code: 400
Response body: Disallowed CORS headers
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/48/repo.py", line 29, in <module>
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Expected 200, got 400
```
</details>

## Why This Is A Bug

The middleware has an inconsistency in how it processes allowed headers versus how it validates them:

1. **During initialization** (line 67 of cors.py): Headers are lowercased but whitespace is preserved:
   ```python
   self.allow_headers = [h.lower() for h in allow_headers]
   ```
   Example: `' X-Custom-Header '` becomes `' x-custom-header '` (spaces retained)

2. **During validation** (lines 128-129 of cors.py): Headers are lowercased AND stripped before checking:
   ```python
   for header in [h.lower() for h in requested_headers.split(",")]:
       if header.strip() not in self.allow_headers:
   ```
   Example: `' x-custom-header '` becomes `'x-custom-header'` (spaces removed) for comparison

This creates a logical inconsistency where any header configured with leading or trailing whitespace can never pass validation, even when the exact same header (including whitespace) is requested. The middleware accepts the configuration without error but then silently fails to match these headers during preflight validation.

While RFC 7230 specifies that HTTP header field names should not contain whitespace, the middleware's behavior is still problematic because:
- It accepts headers with whitespace during configuration without any warning or error
- It processes them inconsistently, making the configuration non-functional
- If headers with whitespace are invalid, they should be rejected consistently, preferably at configuration time with a clear error message

## Relevant Context

The CORS specification (W3C) and HTTP specification (RFC 7230) provide context:
- RFC 7230 Section 3.2 states that header field names are tokens and should not contain whitespace
- The Access-Control-Request-Headers header value should be a comma-separated list of header names
- Starlette's implementation correctly handles spaces around commas in the header list (e.g., "Header1 , Header2")
- The issue is specifically with whitespace within the header names themselves

The bug manifests with any header containing leading/trailing whitespace, including:
- Simple spaces: `' header '`
- Tabs: `'\theader\t'`
- Non-breaking spaces: `'\xa0header\xa0'`
- Even just a comma: `','` (which becomes empty string when stripped)

## Proposed Fix

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