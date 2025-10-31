# Bug Report: starlette.middleware.cors.CORSMiddleware Inconsistent Whitespace Handling in Header Configuration

**Target**: `starlette.middleware.cors.CORSMiddleware`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

CORSMiddleware rejects valid CORS preflight requests when `allow_headers` configuration contains trailing or leading whitespace, due to inconsistent whitespace handling between configuration storage and request validation.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from starlette.datastructures import Headers
from starlette.middleware.cors import CORSMiddleware


@given(
    header_name=st.text(min_size=1, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz-"),
    spaces_before=st.integers(min_value=0, max_value=3),
    spaces_after=st.integers(min_value=0, max_value=3)
)
@settings(max_examples=100)
def test_cors_allow_headers_whitespace(header_name, spaces_before, spaces_after):
    header_with_spaces = " " * spaces_before + header_name + " " * spaces_after

    middleware = CORSMiddleware(
        app=None,
        allow_origins=["*"],
        allow_headers=[header_with_spaces]
    )

    request_headers = Headers({
        "origin": "http://example.com",
        "access-control-request-method": "GET",
        "access-control-request-headers": header_name
    })

    response = middleware.preflight_response(request_headers)

    assert response.status_code == 200, \
        f"Expected 200 OK but got {response.status_code}. " \
        f"Header '{header_with_spaces}' (with spaces) was allowed in config, " \
        f"but request header '{header_name}' (without spaces) was rejected."


# Run the test
if __name__ == "__main__":
    print("Running property-based test for CORS header whitespace handling...")
    try:
        test_cors_allow_headers_whitespace()
        print("All tests passed!")
    except AssertionError as e:
        print(f"Test failed!")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
```

<details>

<summary>
**Failing input**: `header_name='a', spaces_before=0, spaces_after=1`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/24/hypo.py", line 39, in <module>
    test_cors_allow_headers_whitespace()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/24/hypo.py", line 7, in test_cors_allow_headers_whitespace
    header_name=st.text(min_size=1, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz-"),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/24/hypo.py", line 29, in test_cors_allow_headers_whitespace
    assert response.status_code == 200, \
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Expected 200 OK but got 400. Header 'a ' (with spaces) was allowed in config, but request header 'a' (without spaces) was rejected.
Falsifying example: test_cors_allow_headers_whitespace(
    header_name='a',  # or any other generated value
    spaces_before=0,  # or any other generated value
    spaces_after=1,
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/miniconda/lib/python3.13/site-packages/starlette/middleware/cors.py:130
Running property-based test for CORS header whitespace handling...
Test failed!
Error: Expected 200 OK but got 400. Header 'a ' (with spaces) was allowed in config, but request header 'a' (without spaces) was rejected.
```
</details>

## Reproducing the Bug

```python
from starlette.datastructures import Headers
from starlette.middleware.cors import CORSMiddleware

# Create middleware with a header that has trailing whitespace
middleware = CORSMiddleware(
    app=None,
    allow_origins=["*"],
    allow_headers=["X-Custom "]  # Note the trailing space
)

# Create a preflight request with the header (without whitespace)
request_headers = Headers({
    "origin": "http://example.com",
    "access-control-request-method": "GET",
    "access-control-request-headers": "X-Custom"  # No trailing space
})

# Process the preflight request
response = middleware.preflight_response(request_headers)

# Check the response
print(f"Response status code: {response.status_code}")
print(f"Expected: 200")

if response.status_code != 200:
    print(f"ERROR: Request was rejected!")
    print(f"Response body: {response.body.decode()}")
else:
    print("Success: Request was accepted")

# Demonstrate the issue with the simplest case
print("\n--- Simplest failing case ---")
middleware_minimal = CORSMiddleware(
    app=None,
    allow_origins=["*"],
    allow_headers=["a "]  # Single character header with trailing space
)

request_minimal = Headers({
    "origin": "http://example.com",
    "access-control-request-method": "GET",
    "access-control-request-headers": "a"
})

response_minimal = middleware_minimal.preflight_response(request_minimal)
print(f"Minimal case status: {response_minimal.status_code}")
if response_minimal.status_code != 200:
    print(f"Minimal case error: {response_minimal.body.decode()}")
```

<details>

<summary>
CORS preflight request rejected with 400 error due to header whitespace mismatch
</summary>
```
Response status code: 400
Expected: 200
ERROR: Request was rejected!
Response body: Disallowed CORS headers

--- Simplest failing case ---
Minimal case status: 400
Minimal case error: Disallowed CORS headers
```
</details>

## Why This Is A Bug

This violates expected behavior because the middleware handles whitespace inconsistently between configuration and request validation:

1. **Configuration Phase** (line 67 of cors.py): When headers are configured in `__init__`, they are lowercased but NOT stripped of whitespace:
   ```python
   self.allow_headers = [h.lower() for h in allow_headers]
   ```
   This stores headers like `"x-custom "` with trailing spaces intact.

2. **Request Validation Phase** (lines 128-130 of cors.py): When validating incoming preflight requests, headers are split, lowercased, AND stripped:
   ```python
   for header in [h.lower() for h in requested_headers.split(",")]:
       if header.strip() not in self.allow_headers:
           failures.append("headers")
   ```
   This checks if `"x-custom"` (stripped) exists in `["x-custom "]` (not stripped), which fails.

3. **Violates HTTP Specifications**: RFC 7230 clearly states that HTTP header field names cannot contain whitespace. The middleware should either:
   - Strip whitespace consistently in both places (defensive programming)
   - Reject headers with whitespace at configuration time with a clear error

4. **Silent Failure**: The middleware accepts invalid configuration without warning, then mysteriously rejects valid requests, making debugging difficult for developers who accidentally include whitespace.

## Relevant Context

The bug occurs in the starlette CORS middleware, which is responsible for handling Cross-Origin Resource Sharing (CORS) preflight requests. CORS is a critical security mechanism in web applications that controls which origins can access resources.

Key observations:
- The bug only affects preflight OPTIONS requests when `Access-Control-Request-Headers` is present
- Simple CORS requests (without custom headers) are not affected
- The issue is particularly problematic because whitespace in configuration can come from various sources:
  - Copy-paste from documentation or examples
  - Configuration files with formatting
  - Environment variables with trailing spaces
  - Human error when typing header names

The middleware code is located at: `/home/npc/pbt/agentic-pbt/envs/starlette_env/lib/python3.13/site-packages/starlette/middleware/cors.py`

Related HTTP specifications:
- RFC 7230 Section 3.2: Header field names are tokens that cannot contain whitespace
- WHATWG Fetch Standard: Headers should be normalized by trimming whitespace

## Proposed Fix

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