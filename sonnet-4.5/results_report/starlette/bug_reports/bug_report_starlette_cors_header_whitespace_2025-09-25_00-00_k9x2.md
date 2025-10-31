# Bug Report: CORSMiddleware Header Whitespace Handling

**Target**: `starlette.middleware.cors.CORSMiddleware`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

CORSMiddleware incorrectly rejects valid CORS preflight requests when header names in the `allow_headers` configuration contain leading or trailing whitespace, even though the actual request headers are properly formatted without whitespace.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from starlette.datastructures import Headers
from starlette.middleware.cors import CORSMiddleware


@given(
    header_name=st.text(min_size=1, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz-"),
    spaces_before=st.integers(min_value=0, max_value=3),
    spaces_after=st.integers(min_value=0, max_value=3)
)
def test_cors_allow_headers_whitespace(header_name, spaces_before, spaces_after):
    header_with_spaces = " " * spaces_before + header_name + " " * spaces_after

    middleware = CORSMiddleware(
        app=None,
        allow_origins=["*"],
        allow_headers=[header_with_spaces]
    )

    request_headers = Headers({
        "origin": "http://example.com",
        "access-control-request-method": "GET",
        "access-control-request-headers": header_name
    })

    response = middleware.preflight_response(request_headers)

    assert response.status_code == 200, \
        f"Expected 200 OK but got {response.status_code}. " \
        f"Header '{header_with_spaces}' (with spaces) was allowed in config, " \
        f"but request header '{header_name}' (without spaces) was rejected."
```

**Failing input**: `header_name='a'`, `spaces_after=1` (produces `allow_headers=['a ']`)

## Reproducing the Bug

```python
from starlette.datastructures import Headers
from starlette.middleware.cors import CORSMiddleware

middleware = CORSMiddleware(
    app=None,
    allow_origins=["*"],
    allow_headers=["X-Custom "]
)

request_headers = Headers({
    "origin": "http://example.com",
    "access-control-request-method": "GET",
    "access-control-request-headers": "X-Custom"
})

response = middleware.preflight_response(request_headers)

assert response.status_code == 200, f"Got {response.status_code}, expected 200"
```

## Why This Is A Bug

The bug occurs because of inconsistent whitespace handling:

1. In `__init__` (line 67), `allow_headers` are converted to lowercase but **not stripped**:
   ```python
   self.allow_headers = [h.lower() for h in allow_headers]
   ```
   Result: `["x-custom "]` (with trailing space)

2. In `preflight_response` (lines 128-129), requested headers are lowercased, split by comma, then **stripped** before checking:
   ```python
   for header in [h.lower() for h in requested_headers.split(",")]:
       if header.strip() not in self.allow_headers:
   ```
   Result: Checking if `"x-custom"` is in `["x-custom "]` → **False**

This causes valid requests to be rejected with a 400 error when header names in the configuration accidentally contain whitespace.

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