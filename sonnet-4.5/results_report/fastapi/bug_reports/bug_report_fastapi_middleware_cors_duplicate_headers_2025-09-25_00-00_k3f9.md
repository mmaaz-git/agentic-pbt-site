# Bug Report: CORSMiddleware Duplicate Headers

**Target**: `starlette.middleware.cors.CORSMiddleware` (re-exported as `fastapi.middleware.cors.CORSMiddleware`)
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

CORSMiddleware allows duplicate headers in its `allow_headers` list when users provide headers that differ only in case from the safelisted headers (Accept, Accept-Language, Content-Language, Content-Type).

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/fastapi_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings
from starlette.middleware.cors import CORSMiddleware

def dummy_app(scope, receive, send):
    pass

@given(st.lists(st.sampled_from(["Accept", "accept", "ACCEPT", "content-type", "Content-Type", "X-Custom-Header", "x-custom-header"]), max_size=10))
@settings(max_examples=500)
def test_cors_no_duplicate_headers(allow_headers):
    middleware = CORSMiddleware(dummy_app, allow_headers=allow_headers)

    unique_headers = set(middleware.allow_headers)
    assert len(middleware.allow_headers) == len(unique_headers), \
        f"Duplicate headers found! Input: {allow_headers}, Output: {middleware.allow_headers}"

if __name__ == "__main__":
    test_cors_no_duplicate_headers()
```

**Failing input**: `["accept"]`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/fastapi_env/lib/python3.13/site-packages')

from starlette.middleware.cors import CORSMiddleware

def dummy_app(scope, receive, send):
    pass

middleware = CORSMiddleware(dummy_app, allow_headers=["accept"])

print(middleware.allow_headers)

assert len(middleware.allow_headers) == len(set(middleware.allow_headers))
```

Output:
```
['accept', 'accept-language', 'content-language', 'content-type', 'accept']
AssertionError
```

The header `"accept"` appears twice in the list.

## Why This Is A Bug

The issue occurs in the `CORSMiddleware.__init__` method at lines 58 and 67:

1. Line 58 performs a set union and sort:
   ```python
   allow_headers = sorted(SAFELISTED_HEADERS | set(allow_headers))
   ```
   With input `["accept"]`, this produces: `["Accept", "Accept-Language", "Content-Language", "Content-Type", "accept"]`
   (Note: Both "Accept" from SAFELISTED_HEADERS and "accept" from user input are present)

2. Line 67 lowercases all headers:
   ```python
   self.allow_headers = [h.lower() for h in allow_headers]
   ```
   This produces: `["accept", "accept-language", "content-language", "content-type", "accept"]`

The duplicate "accept" violates the expected behavior where `allow_headers` should be a unique list. This can cause inefficiencies in header validation and unexpected behavior in CORS preflight responses.

## Fix

The fix is to perform case-insensitive deduplication. One approach is to lowercase headers before taking the union:

```diff
--- a/starlette/middleware/cors.py
+++ b/starlette/middleware/cors.py
@@ -55,7 +55,8 @@ class CORSMiddleware:
                 "Access-Control-Max-Age": str(max_age),
             }
         )
-        allow_headers = sorted(SAFELISTED_HEADERS | set(allow_headers))
+        safelisted_lower = {h.lower() for h in SAFELISTED_HEADERS}
+        allow_headers = sorted(safelisted_lower | {h.lower() for h in allow_headers})
         if allow_headers and not allow_all_headers:
             preflight_headers["Access-Control-Allow-Headers"] = ", ".join(allow_headers)
         if allow_credentials:
@@ -64,7 +65,7 @@ class CORSMiddleware:
         self.app = app
         self.allow_origins = allow_origins
         self.allow_methods = allow_methods
-        self.allow_headers = [h.lower() for h in allow_headers]
+        self.allow_headers = list(allow_headers)
         self.allow_all_origins = allow_all_origins
         self.allow_all_headers = allow_all_headers
         self.preflight_explicit_allow_origin = preflight_explicit_allow_origin
```