# Bug Report: starlette.middleware.cors Unicode Header Encoding Crash

**Target**: `starlette.middleware.cors.CORSMiddleware`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `CORSMiddleware` crashes with a `UnicodeEncodeError` when `allow_headers` contains non-latin-1 encodable characters. HTTP headers must be latin-1 encodable, but the middleware doesn't validate this constraint when accepting the `allow_headers` parameter.

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
```

**Failing input**: `allowed_headers=['Ā']` (contains non-latin-1 character U+0100)

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/starlette_env/lib/python3.13/site-packages')

from starlette.middleware.cors import CORSMiddleware
from starlette.datastructures import Headers


def dummy_app(scope, receive, send):
    pass


middleware = CORSMiddleware(dummy_app, allow_headers=['Ā'], allow_origins=["*"])

request_headers = Headers({
    "origin": "http://example.com",
    "access-control-request-method": "GET"
})

response = middleware.preflight_response(request_headers=request_headers)
```

This raises:
```
UnicodeEncodeError: 'latin-1' codec can't encode character '\u0100' in position 57: ordinal not in range(256)
```

## Why This Is A Bug

HTTP header names and values must be encodable in latin-1 (ISO-8859-1) according to the HTTP specification. The CORS middleware should either:
1. Validate that `allow_headers` only contains latin-1 encodable strings in `__init__`, or
2. Handle the encoding error gracefully

Currently, it accepts arbitrary Unicode strings and then crashes when trying to construct the response headers. This is a crash bug that occurs with valid user inputs (the middleware's constructor doesn't document or enforce the latin-1 requirement).

## Fix

Add validation in the `__init__` method to reject non-latin-1 encodable headers:

```diff
--- a/starlette/middleware/cors.py
+++ b/starlette/middleware/cors.py
@@ -55,6 +55,11 @@ class CORSMiddleware:
             "Access-Control-Max-Age": str(max_age),
         }
     )
+    # Validate that all headers can be encoded in latin-1 (HTTP requirement)
+    for header in allow_headers:
+        try:
+            header.encode('latin-1')
+        except UnicodeEncodeError:
+            raise ValueError(f"Header name must be latin-1 encodable: {header!r}")
     allow_headers = sorted(SAFELISTED_HEADERS | set(allow_headers))
     if allow_headers and not allow_all_headers:
         preflight_headers["Access-Control-Allow-Headers"] = ", ".join(allow_headers)
```