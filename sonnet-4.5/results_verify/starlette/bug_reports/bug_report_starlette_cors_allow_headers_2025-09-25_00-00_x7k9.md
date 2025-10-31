# Bug Report: CORSMiddleware allow_headers Incorrect Sorting

**Target**: `starlette.middleware.cors.CORSMiddleware`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `CORSMiddleware` class incorrectly sorts the `allow_headers` list by sorting before lowercasing, resulting in headers that are not in alphabetical order after lowercasing.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from starlette.middleware.cors import CORSMiddleware, SAFELISTED_HEADERS


def dummy_asgi_app(scope, receive, send):
    pass


@given(st.lists(st.text(alphabet=st.characters(min_codepoint=65, max_codepoint=90), min_size=1, max_size=20), min_size=1, max_size=10))
def test_allow_headers_sorted_after_lowercase(headers):
    cors = CORSMiddleware(dummy_asgi_app, allow_headers=headers)
    expected_sorted = sorted([h.lower() for h in (SAFELISTED_HEADERS | set(headers))])
    actual = cors.allow_headers
    assert actual == expected_sorted
```

**Failing input**: `headers=['AD']`

## Reproducing the Bug

```python
from starlette.middleware.cors import CORSMiddleware, SAFELISTED_HEADERS


def dummy_asgi_app(scope, receive, send):
    pass


headers = ['AD']
cors = CORSMiddleware(dummy_asgi_app, allow_headers=headers)

print("Actual allow_headers:", cors.allow_headers)
print("Expected:", sorted([h.lower() for h in (SAFELISTED_HEADERS | set(headers))]))

assert cors.allow_headers[0] == 'accept'
```

Output:
```
Actual allow_headers: ['ad', 'accept', 'accept-language', 'content-language', 'content-type']
Expected: ['accept', 'accept-language', 'ad', 'content-language', 'content-type']
AssertionError
```

## Why This Is A Bug

The code in `cors.py` sorts the headers before lowercasing them (line 58), then lowercases the sorted list (line 67). This breaks alphabetical ordering because uppercase letters sort differently than lowercase letters in ASCII (e.g., 'AD' < 'Accept' but 'ad' > 'accept').

The explicit use of `sorted()` on line 58 indicates the intent was to maintain alphabetical order. However, the current implementation violates this intent for headers with mixed case.

## Fix

```diff
--- a/starlette/middleware/cors.py
+++ b/starlette/middleware/cors.py
@@ -55,11 +55,11 @@ class CORSMiddleware:
                 "Access-Control-Max-Age": str(max_age),
             }
         )
-        allow_headers = sorted(SAFELISTED_HEADERS | set(allow_headers))
+        lowercased_headers = [h.lower() for h in (SAFELISTED_HEADERS | set(allow_headers))]
+        allow_headers = sorted(lowercased_headers)
         if allow_headers and not allow_all_headers:
             preflight_headers["Access-Control-Allow-Headers"] = ", ".join(allow_headers)
         if allow_credentials:
             preflight_headers["Access-Control-Allow-Credentials"] = "true"

         self.app = app
@@ -64,7 +64,7 @@ class CORSMiddleware:
         self.allow_origins = allow_origins
         self.allow_methods = allow_methods
-        self.allow_headers = [h.lower() for h in allow_headers]
+        self.allow_headers = allow_headers
         self.allow_all_origins = allow_all_origins
         self.allow_all_headers = allow_all_headers
         self.preflight_explicit_allow_origin = preflight_explicit_allow_origin
```