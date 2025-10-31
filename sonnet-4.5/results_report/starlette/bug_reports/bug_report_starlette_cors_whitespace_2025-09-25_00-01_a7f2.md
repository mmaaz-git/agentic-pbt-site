# Bug Report: CORSMiddleware Header Whitespace Inconsistency

**Target**: `starlette.middleware.cors.CORSMiddleware.__init__`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`CORSMiddleware` lowercases but doesn't strip whitespace from `allow_headers` during initialization, while the validation logic in `preflight_response()` strips whitespace before comparison. This causes asymmetric behavior where headers with trailing/leading spaces in configuration may incorrectly reject valid requests.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st, assume
from starlette.middleware.cors import CORSMiddleware


@given(
    header_name=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="-")),
    leading_space=st.integers(min_value=0, max_value=3),
    trailing_space=st.integers(min_value=1, max_value=3),
)
@settings(max_examples=200)
def test_cors_allow_headers_whitespace_inconsistency(header_name, leading_space, trailing_space):
    assume(header_name.strip() == header_name)
    assume(len(header_name.strip()) > 0)

    header_with_whitespace = " " * leading_space + header_name + " " * trailing_space

    async def dummy_app(scope, receive, send):
        pass

    middleware = CORSMiddleware(
        dummy_app,
        allow_origins=["http://example.com"],
        allow_headers=[header_with_whitespace]
    )

    stored_header = middleware.allow_headers[0]

    assert stored_header.strip() == stored_header, \
        f"Bug: allow_headers contains whitespace: '{stored_header}' (should be '{stored_header.strip()}')"
```

**Failing input**: `CORSMiddleware(app, allow_headers=["X-Custom-Header "])` (note trailing space)

## Reproducing the Bug

```python
from starlette.middleware.cors import CORSMiddleware

async def dummy_app(scope, receive, send):
    pass

middleware = CORSMiddleware(
    dummy_app,
    allow_origins=["http://example.com"],
    allow_headers=["X-Custom-Header ", "Content-Type"]
)

print(middleware.allow_headers)
```

**Output**: `['x-custom-header ', 'accept', 'accept-language', 'content-language', 'content-type']`

Note that `'x-custom-header '` has a trailing space. Later, in `preflight_response()`, when validating request headers:

```python
for header in [h.lower() for h in requested_headers.split(",")]:
    if header.strip() not in self.allow_headers:
        failures.append("headers")
        break
```

The code does `header.strip()` but `self.allow_headers` contains the un-stripped values, causing mismatches.

## Why This Is A Bug

When a developer configures `allow_headers=["X-Custom-Header "]` (accidentally including whitespace), the middleware stores `"x-custom-header "` (lowercase with space). However, when a legitimate request comes in with header `"X-Custom-Header"` (no space), the validation logic does:

1. Lowercase the requested header: `"x-custom-header"`
2. Strip it: `"x-custom-header".strip()` = `"x-custom-header"`
3. Check if `"x-custom-header"` is in `["x-custom-header ", ...]` → **False**, request rejected!

The inconsistency between initialization (lowercase only) and validation (lowercase + strip) causes valid requests to be incorrectly rejected.

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