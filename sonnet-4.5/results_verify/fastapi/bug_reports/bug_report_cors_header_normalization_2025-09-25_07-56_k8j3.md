# Bug Report: CORSMiddleware Header Case-Sensitivity Issue

**Target**: `starlette.middleware.cors.CORSMiddleware`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

CORSMiddleware produces different internal state when configured with headers that differ only in case, violating the HTTP principle that header names are case-insensitive.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from starlette.middleware.cors import CORSMiddleware

def dummy_app(scope, receive, send):
    pass

@given(st.lists(st.text(alphabet=st.characters(whitelist_categories=('Lu', 'Ll')), min_size=1), min_size=1))
def test_cors_allow_headers_case_insensitive(headers):
    lowercased = [h.lower() for h in headers]
    uppercased = [h.upper() for h in headers]

    middleware_lower = CORSMiddleware(dummy_app, allow_headers=lowercased)
    middleware_upper = CORSMiddleware(dummy_app, allow_headers=uppercased)

    assert middleware_lower.allow_headers == middleware_upper.allow_headers
```

**Failing input**: `['A']`

## Reproducing the Bug

```python
from starlette.middleware.cors import CORSMiddleware

def app(scope, receive, send):
    pass

m1 = CORSMiddleware(app, allow_headers=['custom-header'])
m2 = CORSMiddleware(app, allow_headers=['CUSTOM-HEADER'])

print("With 'custom-header':", m1.allow_headers)
print("With 'CUSTOM-HEADER':", m2.allow_headers)
print("Are they equal?", m1.allow_headers == m2.allow_headers)
```

Output:
```
With 'custom-header': ['accept', 'accept-language', 'content-language', 'content-type', 'custom-header']
With 'CUSTOM-HEADER': ['custom-header', 'accept', 'accept-language', 'content-language', 'content-type']
Are they equal? False
```

## Why This Is A Bug

HTTP header names are case-insensitive per RFC 7230. Two CORSMiddleware instances configured with headers differing only in case should have identical internal state. Currently, the sort order differs because sorting happens before case normalization.

In `cors.py` lines 58-67:
1. Line 58: Headers are sorted while still mixed-case
2. Line 67: Headers are lowercased after sorting

This causes 'A' (ASCII 65) to sort before 'Accept', while 'a' (ASCII 97) sorts after 'accept', resulting in different orderings for case variants of the same header.

## Fix

```diff
--- a/starlette/middleware/cors.py
+++ b/starlette/middleware/cors.py
@@ -55,12 +55,12 @@ class CORSMiddleware:
                 "Access-Control-Max-Age": str(max_age),
             }
         )
-        allow_headers = sorted(SAFELISTED_HEADERS | set(allow_headers))
+        combined_headers = {h.lower() for h in SAFELISTED_HEADERS} | {h.lower() for h in allow_headers}
+        allow_headers = sorted(combined_headers)
         if allow_headers and not allow_all_headers:
             preflight_headers["Access-Control-Allow-Headers"] = ", ".join(allow_headers)
         if allow_credentials:
             preflight_headers["Access-Control-Allow-Credentials"] = "true"

         self.app = app
         self.allow_origins = allow_origins
         self.allow_methods = allow_methods
-        self.allow_headers = [h.lower() for h in allow_headers]
+        self.allow_headers = allow_headers
```