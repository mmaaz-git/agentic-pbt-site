# Bug Report: starlette.middleware.cors CORSMiddleware Unicode Case-Folding

**Target**: `starlette.middleware.cors.CORSMiddleware`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The CORS middleware's header matching logic fails for certain Unicode characters that don't round-trip through upper/lower case transformations, causing valid headers to be incorrectly rejected.

## Property-Based Test

```python
from hypothesis import given, settings
import hypothesis.strategies as st
from starlette.middleware.cors import CORSMiddleware


def dummy_app(scope, receive, send):
    pass


@given(st.text(min_size=1, max_size=50))
@settings(max_examples=200)
def test_cors_unicode_case_folding_bug(header):
    middleware = CORSMiddleware(
        dummy_app,
        allow_headers=[header]
    )

    lower_header = header.lower()
    upper_header = header.upper()
    upper_then_lower = upper_header.lower()

    if lower_header in middleware.allow_headers:
        assert upper_then_lower in middleware.allow_headers
```

**Failing input**: `'µ'` (MICRO SIGN U+00B5)

## Reproducing the Bug

```python
from starlette.middleware.cors import CORSMiddleware


def dummy_app(scope, receive, send):
    pass


middleware = CORSMiddleware(
    dummy_app,
    allow_headers=['µ']
)

print("Allowed headers:", middleware.allow_headers)
print("'µ'.upper():", 'µ'.upper())
print("'µ'.upper().lower():", 'µ'.upper().lower())
print()
print("Is 'µ' in allow_headers?", 'µ' in middleware.allow_headers)
print("Is 'Μ'.lower() in allow_headers?", 'Μ'.lower() in middleware.allow_headers)
```

Output:
```
Allowed headers: ['accept', 'accept-language', 'content-language', 'content-type', 'µ']
'µ'.upper(): Μ
'µ'.upper().lower(): μ
Is 'µ' in allow_headers? True
Is 'Μ'.lower() in allow_headers? False
```

## Why This Is A Bug

The CORS middleware stores allowed headers in lowercase form (line 67 in `cors.py`):
```python
self.allow_headers = [h.lower() for h in allow_headers]
```

When validating preflight requests, it lowercases incoming headers and checks if they're in `allow_headers` (lines 128-129):
```python
for header in [h.lower() for h in requested_headers.split(",")]:
    if header.strip() not in self.allow_headers:
```

However, certain Unicode characters don't satisfy the round-trip property `c.lower() == c.upper().lower()`:
- µ (MICRO SIGN U+00B5) uppercases to Μ (GREEK CAPITAL MU U+039C)
- Μ lowercases to μ (GREEK SMALL MU U+03BC)
- µ ≠ μ

This causes a client sending "Μ" (uppercase of µ) to be rejected even though "µ" was explicitly allowed.

Similarly, ß (GERMAN SHARP S) uppercases to "SS", which lowercases to "ss", not ß.

## Fix

Use `casefold()` instead of `lower()` for case-insensitive comparisons. The `casefold()` method is designed specifically for case-insensitive matching and handles these Unicode edge cases correctly.

```diff
--- a/starlette/middleware/cors.py
+++ b/starlette/middleware/cors.py
@@ -64,7 +64,7 @@ class CORSMiddleware:
         self.app = app
         self.allow_origins = allow_origins
         self.allow_methods = allow_methods
-        self.allow_headers = [h.lower() for h in allow_headers]
+        self.allow_headers = [h.casefold() for h in allow_headers]
         self.allow_all_origins = allow_all_origins
         self.allow_all_headers = allow_all_headers
         self.preflight_explicit_allow_origin = preflight_explicit_allow_origin
@@ -125,7 +125,7 @@ class CORSMiddleware:
         if self.allow_all_headers and requested_headers is not None:
             headers["Access-Control-Allow-Headers"] = requested_headers
         elif requested_headers is not None:
-            for header in [h.lower() for h in requested_headers.split(",")]:
+            for header in [h.casefold() for h in requested_headers.split(",")]:
                 if header.strip() not in self.allow_headers:
                     failures.append("headers")
                     break
```