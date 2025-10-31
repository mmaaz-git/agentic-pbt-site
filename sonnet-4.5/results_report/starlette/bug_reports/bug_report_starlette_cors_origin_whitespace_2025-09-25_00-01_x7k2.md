# Bug Report: CORSMiddleware Origin Whitespace Handling

**Target**: `starlette.middleware.cors.CORSMiddleware`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

CORSMiddleware incorrectly rejects valid CORS preflight requests when origin URLs in the `allow_origins` configuration contain leading or trailing whitespace, even though the actual request origins are properly formatted without whitespace.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from starlette.datastructures import Headers
from starlette.middleware.cors import CORSMiddleware


@given(
    origin=st.text(min_size=1, max_size=30, alphabet="abcdefghijklmnopqrstuvwxyz:/.-"),
    spaces_before=st.integers(min_value=0, max_value=3),
    spaces_after=st.integers(min_value=0, max_value=3)
)
def test_cors_allow_origins_whitespace(origin, spaces_before, spaces_after):
    origin_with_spaces = " " * spaces_before + origin + " " * spaces_after

    middleware = CORSMiddleware(
        app=None,
        allow_origins=[origin_with_spaces]
    )

    request_headers = Headers({
        "origin": origin,
        "access-control-request-method": "GET"
    })

    response = middleware.preflight_response(request_headers)

    assert response.status_code == 200, \
        f"Expected 200 OK but got {response.status_code}. " \
        f"Origin '{origin_with_spaces}' (with spaces) was allowed in config, " \
        f"but request origin '{origin}' (without spaces) was rejected."
```

**Failing input**: `origin=':'`, `spaces_after=1` (produces `allow_origins=[': ']`)

## Reproducing the Bug

```python
from starlette.datastructures import Headers
from starlette.middleware.cors import CORSMiddleware

middleware = CORSMiddleware(
    app=None,
    allow_origins=["http://example.com "]
)

request_headers = Headers({
    "origin": "http://example.com",
    "access-control-request-method": "GET"
})

response = middleware.preflight_response(request_headers)

assert response.status_code == 200, f"Got {response.status_code}, expected 200"
```

## Why This Is A Bug

The bug occurs because `allow_origins` values are stored without normalization:

1. In `__init__` (line 65), `allow_origins` are stored as-is:
   ```python
   self.allow_origins = allow_origins
   ```
   Result: `["http://example.com "]` (with trailing space)

2. In `is_allowed_origin` (line 102), the check is a simple membership test:
   ```python
   return origin in self.allow_origins
   ```
   Result: Checking if `"http://example.com"` is in `["http://example.com "]` → **False**

This causes valid requests to be rejected with a 400 error when origin URLs in the configuration accidentally contain whitespace.

## Fix

```diff
--- a/starlette/middleware/cors.py
+++ b/starlette/middleware/cors.py
@@ -62,7 +62,7 @@ class CORSMiddleware:
         self.preflight_headers["Access-Control-Allow-Credentials"] = "true"

     self.app = app
-    self.allow_origins = allow_origins
+    self.allow_origins = [origin.strip() for origin in allow_origins]
     self.allow_methods = allow_methods
     self.allow_headers = [h.lower().strip() for h in allow_headers]
     self.allow_all_origins = allow_all_origins
```