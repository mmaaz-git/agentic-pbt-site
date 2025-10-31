# Bug Report: starlette.middleware.cors Unicode Header Encoding Crash

**Target**: `starlette.middleware.cors.CORSMiddleware`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `CORSMiddleware` crashes with a `UnicodeEncodeError` when `allow_headers` contains non-latin-1 encodable characters, accepting invalid input during initialization that fails when constructing the HTTP response.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/starlette_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, assume
from starlette.middleware.cors import CORSMiddleware
from starlette.datastructures import Headers


def dummy_app(scope, receive, send):
    pass


@given(st.lists(st.text(min_size=1, max_size=20), min_size=0, max_size=5))
def test_cors_whitespace_only_header(allowed_headers):
    assume(all(h.strip() for h in allowed_headers))

    middleware = CORSMiddleware(dummy_app, allow_headers=allowed_headers, allow_origins=["*"])

    request_headers = Headers({
        "origin": "http://example.com",
        "access-control-request-method": "GET",
        "access-control-request-headers": "   "
    })

    response = middleware.preflight_response(request_headers=request_headers)
    assert response.status_code in [200, 400]


if __name__ == "__main__":
    test_cors_whitespace_only_header()
```

<details>

<summary>
**Failing input**: `allowed_headers=['Ā']`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/43/hypo.py", line 30, in <module>
    test_cors_whitespace_only_header()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/43/hypo.py", line 14, in test_cors_whitespace_only_header
    def test_cors_whitespace_only_header(allowed_headers):
                   ^^^
  File "/home/npc/pbt/agentic-pbt/envs/starlette_env/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/43/hypo.py", line 25, in test_cors_whitespace_only_header
    response = middleware.preflight_response(request_headers=request_headers)
  File "/home/npc/pbt/agentic-pbt/envs/starlette_env/lib/python3.13/site-packages/starlette/middleware/cors.py", line 138, in preflight_response
    return PlainTextResponse(failure_text, status_code=400, headers=headers)
  File "/home/npc/pbt/agentic-pbt/envs/starlette_env/lib/python3.13/site-packages/starlette/responses.py", line 48, in __init__
    self.init_headers(headers)
    ~~~~~~~~~~~~~~~~~^^^^^^^^^
  File "/home/npc/pbt/agentic-pbt/envs/starlette_env/lib/python3.13/site-packages/starlette/responses.py", line 63, in init_headers
    raw_headers = [(k.lower().encode("latin-1"), v.encode("latin-1")) for k, v in headers.items()]
                                                 ~~~~~~~~^^^^^^^^^^^
UnicodeEncodeError: 'latin-1' codec can't encode character '\u0100' in position 57: ordinal not in range(256)
Falsifying example: test_cors_whitespace_only_header(
    allowed_headers=['Ā'],
)
```
</details>

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/starlette_env/lib/python3.13/site-packages')

from starlette.middleware.cors import CORSMiddleware
from starlette.datastructures import Headers


def dummy_app(scope, receive, send):
    pass


# Create middleware with non-latin-1 character in allow_headers
middleware = CORSMiddleware(dummy_app, allow_headers=['Ā'], allow_origins=["*"])

# Create request headers for a preflight request
request_headers = Headers({
    "origin": "http://example.com",
    "access-control-request-method": "GET"
})

# This will raise UnicodeEncodeError
response = middleware.preflight_response(request_headers=request_headers)
print(f"Response status: {response.status_code}")
```

<details>

<summary>
UnicodeEncodeError when encoding header 'Ā' (U+0100)
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/43/repo.py", line 22, in <module>
    response = middleware.preflight_response(request_headers=request_headers)
  File "/home/npc/pbt/agentic-pbt/envs/starlette_env/lib/python3.13/site-packages/starlette/middleware/cors.py", line 140, in preflight_response
    return PlainTextResponse("OK", status_code=200, headers=headers)
  File "/home/npc/pbt/agentic-pbt/envs/starlette_env/lib/python3.13/site-packages/starlette/responses.py", line 48, in __init__
    self.init_headers(headers)
    ~~~~~~~~~~~~~~~~~^^^^^^^^^
  File "/home/npc/pbt/agentic-pbt/envs/starlette_env/lib/python3.13/site-packages/starlette/responses.py", line 63, in init_headers
    raw_headers = [(k.lower().encode("latin-1"), v.encode("latin-1")) for k, v in headers.items()]
                                                 ~~~~~~~~^^^^^^^^^^^
UnicodeEncodeError: 'latin-1' codec can't encode character '\u0100' in position 57: ordinal not in range(256)
```
</details>

## Why This Is A Bug

This violates expected behavior because the `CORSMiddleware.__init__` method accepts any Unicode string in the `allow_headers` parameter without validation, but these values are later used to construct HTTP response headers which must be latin-1 encodable per RFC 7230. The bug manifests as:

1. **Deferred failure**: The middleware accepts invalid input during initialization (line 58-60 in cors.py) without any validation
2. **Cryptic error**: The crash occurs later with an unclear `UnicodeEncodeError` when creating the response (responses.py:63)
3. **Undocumented constraint**: The Starlette documentation describes `allow_headers` as "A list of HTTP request headers" without mentioning the latin-1 encoding requirement

The character 'Ā' (U+0100, Latin Extended-A) cannot be encoded in latin-1 (ISO-8859-1) which only supports characters U+0000 to U+00FF. When the middleware adds this character to the "Access-Control-Allow-Headers" response header and PlainTextResponse tries to encode it, the operation fails.

## Relevant Context

The crash occurs in the response construction path:
1. `CORSMiddleware.__init__` (cors.py:58-60) stores headers in `preflight_headers` dict without validation
2. `CORSMiddleware.preflight_response()` (cors.py:140) passes these headers to `PlainTextResponse`
3. `Response.init_headers()` (responses.py:63) attempts to encode all headers as latin-1
4. The encoding fails for characters outside the latin-1 range (ordinal > 255)

HTTP/1.1 specification (RFC 7230 Section 3.2) states that header field values are historically limited to ISO-8859-1 (latin-1) charset. Modern HTTP recommends restricting to US-ASCII for new header fields.

Related code locations:
- Header storage: `/starlette/middleware/cors.py:60`
- Response creation: `/starlette/middleware/cors.py:138-140`
- Encoding failure: `/starlette/responses.py:63`

## Proposed Fix

```diff
--- a/starlette/middleware/cors.py
+++ b/starlette/middleware/cors.py
@@ -55,6 +55,14 @@ class CORSMiddleware:
                 "Access-Control-Max-Age": str(max_age),
             }
         )
+
+        # Validate that all headers can be encoded in latin-1 (HTTP requirement)
+        for header in allow_headers:
+            try:
+                header.encode('latin-1')
+            except UnicodeEncodeError:
+                raise ValueError(f"Header name '{header}' contains non-latin-1 characters which are not valid in HTTP headers")
+
         allow_headers = sorted(SAFELISTED_HEADERS | set(allow_headers))
         if allow_headers and not allow_all_headers:
             preflight_headers["Access-Control-Allow-Headers"] = ", ".join(allow_headers)
```