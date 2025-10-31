# Bug Report: CORSMiddleware Header Case Sensitivity in Sorting

**Target**: `starlette.middleware.cors.CORSMiddleware`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

CORSMiddleware produces different `allow_headers` lists when the same header is provided with different capitalization, violating the HTTP specification that headers are case-insensitive.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from starlette.middleware.cors import CORSMiddleware


@given(
    header=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("Lu", "Ll"), whitelist_characters="-"))
)
def test_cors_headers_case_insensitive_property(header):
    middleware_upper = CORSMiddleware(None, allow_headers=[header.upper()])
    middleware_lower = CORSMiddleware(None, allow_headers=[header.lower()])

    assert middleware_upper.allow_headers == middleware_lower.allow_headers
```

**Failing input**: `header='A'`

## Reproducing the Bug

```python
from starlette.middleware.cors import CORSMiddleware

middleware_upper = CORSMiddleware(None, allow_headers=['A'])
middleware_lower = CORSMiddleware(None, allow_headers=['a'])

print("With 'A':", middleware_upper.allow_headers)
print("With 'a':", middleware_lower.allow_headers)
print("Equal?", middleware_upper.allow_headers == middleware_lower.allow_headers)
```

**Output:**
```
With 'A': ['a', 'accept', 'accept-language', 'content-language', 'content-type']
With 'a': ['accept', 'accept-language', 'content-language', 'content-type', 'a']
Equal? False
```

## Why This Is A Bug

HTTP headers are case-insensitive according to RFC 7230. The CORSMiddleware correctly lowercases all headers for comparison (line 67), but the sorting happens before lowercasing (line 58), which creates an inconsistent state.

When `allow_headers=['A']` is provided:
1. Line 58: `sorted({'Accept', 'Accept-Language', 'Content-Language', 'Content-Type', 'A'})` → uppercase 'A' sorts before other headers
2. Line 67: Lowercase all → `['a', 'accept', 'accept-language', 'content-language', 'content-type']`

When `allow_headers=['a']` is provided:
1. Line 58: `sorted({'Accept', 'Accept-Language', 'Content-Language', 'Content-Type', 'a'})` → lowercase 'a' sorts after capitalized headers
2. Line 67: Lowercase all → `['accept', 'accept-language', 'content-language', 'content-type', 'a']`

This violates the metamorphic property that `CORSMiddleware(allow_headers=['A'])` should be equivalent to `CORSMiddleware(allow_headers=['a'])`.

## Fix

```diff
--- a/starlette/middleware/cors.py
+++ b/starlette/middleware/cors.py
@@ -55,9 +55,9 @@ class CORSMiddleware:
             }
         )
-        allow_headers = sorted(SAFELISTED_HEADERS | set(allow_headers))
+        lowercased_allow_headers = {h.lower() for h in allow_headers}
+        lowercased_safelisted = {h.lower() for h in SAFELISTED_HEADERS}
+        allow_headers = sorted(lowercased_safelisted | lowercased_allow_headers)
         if allow_headers and not allow_all_headers:
             preflight_headers["Access-Control-Allow-Headers"] = ", ".join(allow_headers)
         if allow_credentials:
             preflight_headers["Access-Control-Allow-Credentials"] = "true"

         self.app = app
         self.allow_origins = allow_origins
         self.allow_methods = allow_methods
-        self.allow_headers = [h.lower() for h in allow_headers]
+        self.allow_headers = allow_headers
         self.allow_all_origins = allow_all_origins
         self.allow_all_headers = allow_all_headers
```