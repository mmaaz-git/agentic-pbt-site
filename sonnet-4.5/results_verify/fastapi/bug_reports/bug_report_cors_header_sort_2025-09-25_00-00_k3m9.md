# Bug Report: CORSMiddleware Header Sorting

**Target**: `starlette.middleware.cors.CORSMiddleware`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `allow_headers` list in CORSMiddleware is not sorted after lowercasing, violating the code's intent to maintain sorted order.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from starlette.middleware.cors import CORSMiddleware

@given(st.lists(st.text(min_size=1)))
@settings(max_examples=500)
def test_cors_allow_headers_sorted_and_lowercased(headers):
    middleware = CORSMiddleware(
        app=lambda scope, receive, send: None,
        allow_headers=headers
    )

    stored = middleware.allow_headers

    assert stored == sorted(stored)
```

**Failing input**: `headers=['[']`

## Reproducing the Bug

```python
from starlette.middleware.cors import CORSMiddleware

middleware = CORSMiddleware(
    app=lambda scope, receive, send: None,
    allow_headers=['[']
)

print(middleware.allow_headers)
print(sorted(middleware.allow_headers))
assert middleware.allow_headers == sorted(middleware.allow_headers)
```

Output:
```
['accept', 'accept-language', 'content-language', 'content-type', '[']
['[', 'accept', 'accept-language', 'content-language', 'content-type']
AssertionError
```

## Why This Is A Bug

In `cors.py` line 58, the code explicitly sorts the headers, indicating intent to maintain sorted order:
```python
allow_headers = sorted(SAFELISTED_HEADERS | set(allow_headers))
```

However, line 67 lowercases the headers AFTER sorting:
```python
self.allow_headers = [h.lower() for h in allow_headers]
```

This breaks the sort order because lowercasing can change the relative order of strings (e.g., 'Z' < 'a' in ASCII, but 'z' > 'a').

## Fix

```diff
--- a/starlette/middleware/cors.py
+++ b/starlette/middleware/cors.py
@@ -55,7 +55,6 @@
                 "Access-Control-Max-Age": str(max_age),
             }
         )
-        allow_headers = sorted(SAFELISTED_HEADERS | set(allow_headers))
         if allow_headers and not allow_all_headers:
             preflight_headers["Access-Control-Allow-Headers"] = ", ".join(allow_headers)
         if allow_credentials:
@@ -64,7 +63,7 @@
         self.app = app
         self.allow_origins = allow_origins
         self.allow_methods = allow_methods
-        self.allow_headers = [h.lower() for h in allow_headers]
+        self.allow_headers = sorted([h.lower() for h in (SAFELISTED_HEADERS | set(allow_headers))])
         self.allow_all_origins = allow_all_origins
         self.allow_all_headers = allow_all_headers
         self.preflight_explicit_allow_origin = preflight_explicit_allow_origin
```