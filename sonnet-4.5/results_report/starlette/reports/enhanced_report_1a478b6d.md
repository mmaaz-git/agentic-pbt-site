# Bug Report: starlette.middleware.cors.CORSMiddleware Method Whitespace Handling

**Target**: `starlette.middleware.cors.CORSMiddleware`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

CORSMiddleware incorrectly rejects valid CORS preflight requests when HTTP methods in the `allow_methods` configuration contain leading or trailing whitespace, causing legitimate cross-origin requests to fail with a 400 error.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from starlette.datastructures import Headers
from starlette.middleware.cors import CORSMiddleware


@given(
    method=st.sampled_from(["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"]),
    spaces_before=st.integers(min_value=0, max_value=3),
    spaces_after=st.integers(min_value=0, max_value=3)
)
@settings(max_examples=50)
def test_cors_allow_methods_whitespace(method, spaces_before, spaces_after):
    method_with_spaces = " " * spaces_before + method + " " * spaces_after

    middleware = CORSMiddleware(
        app=None,
        allow_origins=["*"],
        allow_methods=[method_with_spaces]
    )

    request_headers = Headers({
        "origin": "http://example.com",
        "access-control-request-method": method
    })

    response = middleware.preflight_response(request_headers)

    assert response.status_code == 200, \
        f"Expected 200 OK but got {response.status_code}. " \
        f"Method '{method_with_spaces}' (with spaces) was allowed in config, " \
        f"but request method '{method}' (without spaces) was rejected."

if __name__ == "__main__":
    test_cors_allow_methods_whitespace()
```

<details>

<summary>
**Failing input**: `method='GET', spaces_before=0, spaces_after=1`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/47/hypo.py", line 34, in <module>
    test_cors_allow_methods_whitespace()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/47/hypo.py", line 7, in test_cors_allow_methods_whitespace
    method=st.sampled_from(["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"]),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/47/hypo.py", line 28, in test_cors_allow_methods_whitespace
    assert response.status_code == 200, \
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Expected 200 OK but got 400. Method 'GET ' (with spaces) was allowed in config, but request method 'GET' (without spaces) was rejected.
Falsifying example: test_cors_allow_methods_whitespace(
    method='GET',  # or any other generated value
    spaces_before=0,  # or any other generated value
    spaces_after=1,
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/miniconda/lib/python3.13/site-packages/starlette/middleware/cors.py:121
```
</details>

## Reproducing the Bug

```python
from starlette.datastructures import Headers
from starlette.middleware.cors import CORSMiddleware

# Test case: method with trailing space in config
middleware = CORSMiddleware(
    app=None,
    allow_origins=["*"],
    allow_methods=["GET "]  # Note the trailing space
)

request_headers = Headers({
    "origin": "http://example.com",
    "access-control-request-method": "GET"  # No trailing space
})

response = middleware.preflight_response(request_headers)

print(f"Configuration: allow_methods=['GET '] (with trailing space)")
print(f"Request: access-control-request-method='GET' (no space)")
print(f"Response status code: {response.status_code}")
print(f"Expected: 200, Got: {response.status_code}")

if response.status_code != 200:
    print("\nERROR: Valid CORS preflight request was rejected!")
    print("The middleware failed to match 'GET' request with 'GET ' in allow_methods")
```

<details>

<summary>
CORS preflight request failed with 400 error
</summary>
```
Configuration: allow_methods=['GET '] (with trailing space)
Request: access-control-request-method='GET' (no space)
Response status code: 400
Expected: 200, Got: 400

ERROR: Valid CORS preflight request was rejected!
The middleware failed to match 'GET' request with 'GET ' in allow_methods
```
</details>

## Why This Is A Bug

This violates expected behavior because HTTP method names are well-defined tokens that never include whitespace. Browsers always send clean method names like "GET", never " GET " or "GET ". When a developer accidentally includes whitespace in the `allow_methods` configuration (which commonly happens when loading from environment variables or configuration files), the middleware silently rejects all legitimate CORS preflight requests for that method.

The bug demonstrates an inconsistency within the same class:
- Headers ARE normalized: line 67 lowercases them (`self.allow_headers = [h.lower() for h in allow_headers]`) and line 129 strips whitespace when checking (`if header.strip() not in self.allow_headers`)
- Methods are NOT normalized: line 66 stores them as-is (`self.allow_methods = allow_methods`) and line 120 does direct membership testing (`if requested_method not in self.allow_methods`)

This inconsistency makes the middleware fragile and error-prone, especially since the failure mode is silent - CORS requests simply fail with no clear indication that trailing whitespace in configuration is the cause.

## Relevant Context

The bug occurs in `/home/npc/pbt/agentic-pbt/envs/starlette_env/lib/python3.13/site-packages/starlette/middleware/cors.py`:
- Line 66: Methods are stored without normalization
- Line 120: Direct string membership test fails when configuration has whitespace
- Contrast with line 129: Headers ARE stripped during comparison

The Starlette documentation doesn't specify that method names must be exact matches without whitespace, nor does it warn about this behavior. Since HTTP specifications define method names as tokens without whitespace, and browsers will never send whitespace in method names, the current behavior creates a trap for developers who may inadvertently include whitespace in their configuration.

Common scenarios where this bug manifests:
1. Configuration loaded from YAML/JSON files where trailing spaces are invisible
2. Environment variables that accidentally include whitespace
3. Copy-paste errors when configuring allowed methods

## Proposed Fix

```diff
--- a/starlette/middleware/cors.py
+++ b/starlette/middleware/cors.py
@@ -63,7 +63,7 @@ class CORSMiddleware:

         self.app = app
         self.allow_origins = allow_origins
-        self.allow_methods = allow_methods
+        self.allow_methods = [method.strip() for method in allow_methods]
         self.allow_headers = [h.lower() for h in allow_headers]
         self.allow_all_origins = allow_all_origins
         self.allow_all_headers = allow_all_headers
```