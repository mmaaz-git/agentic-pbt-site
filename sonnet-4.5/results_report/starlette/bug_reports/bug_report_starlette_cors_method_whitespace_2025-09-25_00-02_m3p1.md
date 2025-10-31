# Bug Report: CORSMiddleware Method Whitespace Handling

**Target**: `starlette.middleware.cors.CORSMiddleware`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

CORSMiddleware incorrectly rejects valid CORS preflight requests when HTTP methods in the `allow_methods` configuration contain leading or trailing whitespace, even though the actual request methods are properly formatted without whitespace.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from starlette.datastructures import Headers
from starlette.middleware.cors import CORSMiddleware


@given(
    method=st.sampled_from(["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"]),
    spaces_before=st.integers(min_value=0, max_value=3),
    spaces_after=st.integers(min_value=0, max_value=3)
)
def test_cors_allow_methods_whitespace(method, spaces_before, spaces_after):
    method_with_spaces = " " * spaces_before + method + " " * spaces_after

    middleware = CORSMiddleware(
        app=None,
        allow_origins=["*"],
        allow_methods=[method_with_spaces]
    )

    request_headers = Headers({
        "origin": "http://example.com",
        "access-control-request-method": method
    })

    response = middleware.preflight_response(request_headers)

    assert response.status_code == 200, \
        f"Expected 200 OK but got {response.status_code}. " \
        f"Method '{method_with_spaces}' (with spaces) was allowed in config, " \
        f"but request method '{method}' (without spaces) was rejected."
```

**Failing input**: `method='GET'`, `spaces_after=1` (produces `allow_methods=['GET ']`)

## Reproducing the Bug

```python
from starlette.datastructures import Headers
from starlette.middleware.cors import CORSMiddleware

middleware = CORSMiddleware(
    app=None,
    allow_origins=["*"],
    allow_methods=["GET "]
)

request_headers = Headers({
    "origin": "http://example.com",
    "access-control-request-method": "GET"
})

response = middleware.preflight_response(request_headers)

assert response.status_code == 200, f"Got {response.status_code}, expected 200"
```

## Why This Is A Bug

The bug occurs because `allow_methods` values are stored without normalization:

1. In `__init__` (line 66), `allow_methods` are stored as-is (after wildcard expansion):
   ```python
   self.allow_methods = allow_methods
   ```
   Result: `["GET "]` (with trailing space)

2. In `preflight_response` (line 120), the check is a simple membership test:
   ```python
   if requested_method not in self.allow_methods:
       failures.append("method")
   ```
   Result: Checking if `"GET"` is in `["GET "]` → **False**

This causes valid requests to be rejected with a 400 error when HTTP methods in the configuration accidentally contain whitespace.

## Fix

```diff
--- a/starlette/middleware/cors.py
+++ b/starlette/middleware/cors.py
@@ -63,7 +63,7 @@ class CORSMiddleware:

     self.app = app
     self.allow_origins = [origin.strip() for origin in allow_origins]
-    self.allow_methods = allow_methods
+    self.allow_methods = [method.strip() for method in allow_methods]
     self.allow_headers = [h.lower().strip() for h in allow_headers]
     self.allow_all_origins = allow_all_origins
     self.allow_all_headers = allow_all_headers
```