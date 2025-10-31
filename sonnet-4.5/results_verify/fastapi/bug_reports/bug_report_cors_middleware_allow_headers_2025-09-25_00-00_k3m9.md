# Bug Report: CORSMiddleware allow_headers Not Sorted and Contains Duplicates

**Target**: `starlette.middleware.cors.CORSMiddleware.__init__`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

CORSMiddleware's `allow_headers` attribute can contain duplicates and is not properly sorted when user-provided headers differ only in case. The code attempts to deduplicate via `set()` and sort via `sorted()`, but these operations occur before case normalization, causing both to fail.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from starlette.middleware.cors import CORSMiddleware

@given(st.lists(st.text(alphabet=st.characters(min_codepoint=33, max_codepoint=126), min_size=1, max_size=20)))
@settings(max_examples=1000)
def test_cors_allow_headers_no_duplicates(allow_headers):
    async def dummy_app(scope, receive, send):
        pass

    middleware = CORSMiddleware(app=dummy_app, allow_headers=allow_headers)
    assert len(middleware.allow_headers) == len(set(middleware.allow_headers))

@given(st.lists(st.text(alphabet=st.characters(min_codepoint=33, max_codepoint=126), min_size=1, max_size=20)))
@settings(max_examples=1000)
def test_cors_allow_headers_sorted(allow_headers):
    async def dummy_app(scope, receive, send):
        pass

    middleware = CORSMiddleware(app=dummy_app, allow_headers=allow_headers)
    assert middleware.allow_headers == sorted(middleware.allow_headers)
```

**Failing inputs**:
- Duplicates bug: `allow_headers=['Q', 'q']`
- Sorting bug: `allow_headers=['[']`

## Reproducing the Bug

```python
from starlette.middleware.cors import CORSMiddleware

async def dummy_app(scope, receive, send):
    pass

middleware = CORSMiddleware(app=dummy_app, allow_headers=['Q', 'q'])
print("Duplicates:", middleware.allow_headers)
print("Has duplicates:", len(middleware.allow_headers) != len(set(middleware.allow_headers)))

middleware2 = CORSMiddleware(app=dummy_app, allow_headers=['['])
print("\nSorting:", middleware2.allow_headers)
print("Is sorted:", middleware2.allow_headers == sorted(middleware2.allow_headers))
```

Output:
```
Duplicates: ['accept', 'accept-language', 'content-language', 'content-type', 'q', 'q']
Has duplicates: True

Sorting: ['accept', 'accept-language', 'content-language', 'content-type', '[']
Is sorted: False
```

## Why This Is A Bug

The code on line 58 of `cors.py` explicitly uses `set()` to deduplicate and `sorted()` to sort headers:
```python
allow_headers = sorted(SAFELISTED_HEADERS | set(allow_headers))
```

However, line 67 lowercases headers AFTER these operations:
```python
self.allow_headers = [h.lower() for h in allow_headers]
```

This causes:
1. **Duplicates**: Headers that differ only in case (e.g., 'Q' and 'q') are not deduplicated because they're different before lowercasing
2. **Incorrect sorting**: Sorting happens on mixed-case strings, but the final result is lowercase, breaking alphabetical order

The code's clear intent (via `set()` and `sorted()`) is violated by the ordering of operations.

## Fix

Apply lowercasing before deduplication and sorting:

```diff
--- a/starlette/middleware/cors.py
+++ b/starlette/middleware/cors.py
@@ -55,7 +55,7 @@ class CORSMiddleware:
                 "Access-Control-Max-Age": str(max_age),
             }
         )
-        allow_headers = sorted(SAFELISTED_HEADERS | set(allow_headers))
+        allow_headers = sorted({h.lower() for h in SAFELISTED_HEADERS} | {h.lower() for h in allow_headers})
         if allow_headers and not allow_all_headers:
             preflight_headers["Access-Control-Allow-Headers"] = ", ".join(allow_headers)
         if allow_credentials:
@@ -64,7 +64,7 @@ class CORSMiddleware:
         self.app = app
         self.allow_origins = allow_origins
         self.allow_methods = allow_methods
-        self.allow_headers = [h.lower() for h in allow_headers]
+        self.allow_headers = list(allow_headers)
         self.allow_all_origins = allow_all_origins
         self.allow_all_headers = allow_all_headers
         self.preflight_explicit_allow_origin = preflight_explicit_allow_origin
```