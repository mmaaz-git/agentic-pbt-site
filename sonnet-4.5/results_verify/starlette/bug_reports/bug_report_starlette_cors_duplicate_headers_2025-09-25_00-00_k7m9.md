# Bug Report: CORSMiddleware Duplicate Headers

**Target**: `starlette.middleware.cors.CORSMiddleware`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

CORSMiddleware creates duplicate headers in `self.allow_headers` when users provide headers that match safelisted headers with different casing (e.g., user provides "accept" when "Accept" is in SAFELISTED_HEADERS).

## Property-Based Test

```python
from hypothesis import given, strategies as st
from starlette.middleware.cors import CORSMiddleware, SAFELISTED_HEADERS


def dummy_app(scope, receive, send):
    pass


@given(st.lists(st.sampled_from(list(SAFELISTED_HEADERS)), min_size=1, max_size=3))
def test_cors_no_duplicate_headers_when_user_provides_safelisted(safelisted_subset):
    user_headers = [h.lower() for h in safelisted_subset]

    cors = CORSMiddleware(
        dummy_app,
        allow_origins=["*"],
        allow_headers=user_headers,
        allow_methods=["GET"]
    )

    assert len(cors.allow_headers) == len(set(cors.allow_headers)), \
        f"Duplicate headers: {cors.allow_headers}"
```

**Failing input**: `safelisted_subset=['Accept']` (or any safelisted header)

## Reproducing the Bug

```python
from starlette.middleware.cors import CORSMiddleware


def dummy_app(scope, receive, send):
    pass


cors = CORSMiddleware(
    dummy_app,
    allow_origins=["*"],
    allow_headers=["accept"],
    allow_methods=["GET"]
)

print("allow_headers:", cors.allow_headers)
print("Expected: ['accept', 'accept-language', 'content-language', 'content-type']")
print("Actual has duplicate 'accept':", cors.allow_headers)
assert len(cors.allow_headers) == len(set(cors.allow_headers))
```

## Why This Is A Bug

The CORS middleware merges user-provided headers with SAFELISTED_HEADERS (which are title-cased), sorts them, then lowercases the result. When a user provides a lowercase version of a safelisted header (e.g., "accept"), the set union treats "Accept" and "accept" as different strings. After lowercasing, both become "accept", creating a duplicate in the final list.

While this doesn't break functionality (the `in` check on line 129 still works), it wastes memory, is inefficient for membership checks, and represents unexpected behavior.

## Fix

```diff
--- a/starlette/middleware/cors.py
+++ b/starlette/middleware/cors.py
@@ -55,7 +55,8 @@ class CORSMiddleware:
                 "Access-Control-Max-Age": str(max_age),
             }
         )
-        allow_headers = sorted(SAFELISTED_HEADERS | set(allow_headers))
+        # Lowercase before merging to avoid case-insensitive duplicates
+        allow_headers = sorted(set(h.lower() for h in SAFELISTED_HEADERS) | set(h.lower() for h in allow_headers))
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