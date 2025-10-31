# Bug Report: Starlette CORS Middleware Empty Header Rejection

**Target**: `starlette.middleware.cors.CORSMiddleware.preflight_response`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The CORS middleware incorrectly rejects valid preflight requests when the `Access-Control-Request-Headers` header contains empty values (empty string, trailing commas, or multiple consecutive commas). This causes legitimate CORS preflight requests to fail with a 400 Bad Request response.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from starlette.middleware.cors import CORSMiddleware
from starlette.datastructures import Headers


def test_cors_preflight_empty_string_header():
    middleware = CORSMiddleware(
        app=lambda: None,
        allow_origins=["https://example.com"],
        allow_headers=["content-type"]
    )

    headers_dict = {
        "origin": "https://example.com",
        "access-control-request-method": "GET",
        "access-control-request-headers": "",
    }
    headers = Headers(raw=[(k.encode(), v.encode()) for k, v in headers_dict.items()])

    response = middleware.preflight_response(headers)

    assert response.status_code == 200
```

**Failing input**: `access-control-request-headers: ""`

## Reproducing the Bug

```python
from starlette.middleware.cors import CORSMiddleware
from starlette.datastructures import Headers

middleware = CORSMiddleware(
    app=lambda: None,
    allow_origins=["https://example.com"],
    allow_headers=["content-type"]
)

headers = Headers(raw=[
    (b"origin", b"https://example.com"),
    (b"access-control-request-method", b"GET"),
    (b"access-control-request-headers", b"")
])

response = middleware.preflight_response(headers)

print(f"Status: {response.status_code}")
print(f"Body: {response.body}")
```

Output:
```
Status: 400
Body: b'Disallowed CORS headers'
```

Other failing cases:
- `access-control-request-headers: ",,,"` → 400 Bad Request
- `access-control-request-headers: "Content-Type,,"` → 400 Bad Request
- `access-control-request-headers: "Content-Type,"` → 400 Bad Request

## Why This Is A Bug

According to the CORS specification, an empty or whitespace-only `Access-Control-Request-Headers` should be treated as no headers being requested, not as a validation failure. Additionally, trailing commas and multiple commas should be handled gracefully by filtering out empty values.

The bug occurs in `starlette/middleware/cors.py` lines 127-131:

```python
elif requested_headers is not None:
    for header in [h.lower() for h in requested_headers.split(",")]:
        if header.strip() not in self.allow_headers:
            failures.append("headers")
            break
```

When `requested_headers` is `""`, `split(",")` returns `[""]`, and `"".strip()` is `""`, which is not in `self.allow_headers`, causing a false rejection.

## Fix

```diff
--- a/starlette/middleware/cors.py
+++ b/starlette/middleware/cors.py
@@ -125,7 +125,7 @@ class CORSMiddleware:
         if self.allow_all_headers and requested_headers is not None:
             headers["Access-Control-Allow-Headers"] = requested_headers
         elif requested_headers is not None:
-            for header in [h.lower() for h in requested_headers.split(",")]:
+            for header in [h.lower().strip() for h in requested_headers.split(",") if h.strip()]:
                 if header.strip() not in self.allow_headers:
                     failures.append("headers")
                     break
```

Alternative fix (cleaner):
```diff
--- a/starlette/middleware/cors.py
+++ b/starlette/middleware/cors.py
@@ -125,8 +125,9 @@ class CORSMiddleware:
         if self.allow_all_headers and requested_headers is not None:
             headers["Access-Control-Allow-Headers"] = requested_headers
         elif requested_headers is not None:
-            for header in [h.lower() for h in requested_headers.split(",")]:
-                if header.strip() not in self.allow_headers:
+            requested_header_list = [h.strip().lower() for h in requested_headers.split(",") if h.strip()]
+            for header in requested_header_list:
+                if header not in self.allow_headers:
                     failures.append("headers")
                     break
```

The fix filters out empty strings before validation by adding `if h.strip()` to the list comprehension, and performs strip() and lower() only once per header.