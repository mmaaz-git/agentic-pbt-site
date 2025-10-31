# Bug Report: starlette.middleware.cors.CORSMiddleware Header Whitespace Inconsistency

**Target**: `starlette.middleware.cors.CORSMiddleware.__init__`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

CORSMiddleware lowercases headers but doesn't strip whitespace during initialization (line 67), while the validation logic strips whitespace before comparison (line 129), causing valid requests to be incorrectly rejected when configuration contains whitespace.

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

if __name__ == "__main__":
    test_cors_allow_headers_whitespace_inconsistency()
```

<details>

<summary>
**Failing input**: `header_name='0', leading_space=0, trailing_space=1`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/39/hypo.py", line 32, in <module>
    test_cors_allow_headers_whitespace_inconsistency()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/39/hypo.py", line 6, in test_cors_allow_headers_whitespace_inconsistency
    header_name=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="-")),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/39/hypo.py", line 28, in test_cors_allow_headers_whitespace_inconsistency
    assert stored_header.strip() == stored_header, \
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Bug: allow_headers contains whitespace: '0 ' (should be '0')
Falsifying example: test_cors_allow_headers_whitespace_inconsistency(
    # The test sometimes passed when commented parts were varied together.
    header_name='0',
    leading_space=0,  # or any other generated value
    trailing_space=1,  # or any other generated value
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/worker_/39/hypo.py:29
```
</details>

## Reproducing the Bug

```python
from starlette.middleware.cors import CORSMiddleware
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.testclient import TestClient

# Create a simple async app
async def dummy_app(scope, receive, send):
    response = JSONResponse({"message": "Hello"})
    await response(scope, receive, send)

# Configure CORSMiddleware with a header that has trailing whitespace
middleware = CORSMiddleware(
    dummy_app,
    allow_origins=["http://example.com"],
    allow_headers=["X-Custom-Header ", "Content-Type"]  # Note the trailing space in X-Custom-Header
)

# Check what headers are stored internally
print("Stored allow_headers:", middleware.allow_headers)
print()

# Now let's test what happens with a preflight request
app = Starlette()

@app.route("/")
async def homepage(request):
    return JSONResponse({"message": "Hello"})

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://example.com"],
    allow_headers=["X-Custom-Header ", "Content-Type"]  # Note the trailing space
)

client = TestClient(app)

# Test a preflight request with the header (without trailing space)
print("Testing preflight request with 'X-Custom-Header' (no trailing space):")
response = client.options(
    "/",
    headers={
        "Origin": "http://example.com",
        "Access-Control-Request-Method": "GET",
        "Access-Control-Request-Headers": "X-Custom-Header"  # No trailing space
    }
)
print(f"Status Code: {response.status_code}")
print(f"Response: {response.text}")
print()

# Test a preflight request with the header (with trailing space to match config)
print("Testing preflight request with 'X-Custom-Header ' (with trailing space):")
response = client.options(
    "/",
    headers={
        "Origin": "http://example.com",
        "Access-Control-Request-Method": "GET",
        "Access-Control-Request-Headers": "X-Custom-Header "  # With trailing space
    }
)
print(f"Status Code: {response.status_code}")
print(f"Response: {response.text}")
```

<details>

<summary>
Output demonstrates valid requests being rejected due to whitespace handling
</summary>
```
Stored allow_headers: ['accept', 'accept-language', 'content-language', 'content-type', 'x-custom-header ']

Testing preflight request with 'X-Custom-Header' (no trailing space):
Status Code: 400
Response: Disallowed CORS headers

Testing preflight request with 'X-Custom-Header ' (with trailing space):
Status Code: 400
Response: Disallowed CORS headers
```
</details>

## Why This Is A Bug

This violates expected behavior because there's an inconsistency in how the middleware processes headers between initialization and validation:

1. **During initialization** (line 67): Headers are lowercased but whitespace is preserved
   - `self.allow_headers = [h.lower() for h in allow_headers]`
   - Input: `["X-Custom-Header "]` → Stored: `["x-custom-header "]`

2. **During validation** (line 129): Headers are stripped before checking membership
   - `if header.strip() not in self.allow_headers:`
   - Checks if `"x-custom-header"` (stripped) is in `["x-custom-header "]` (not stripped) → False

This asymmetry means that when a developer accidentally includes whitespace in their configuration (e.g., `allow_headers=["X-Custom-Header "]`), the middleware silently accepts this configuration but then rejects all legitimate requests for that header. The error message "Disallowed CORS headers" gives no indication that whitespace is the issue, making this bug difficult to debug.

The validation logic's stripping behavior (line 129) suggests the intended design is to be lenient with whitespace, but the initialization code fails to implement this same normalization.

## Relevant Context

The bug is located in `/home/npc/pbt/agentic-pbt/envs/starlette_env/lib/python3.13/site-packages/starlette/middleware/cors.py`:

- **Line 58**: Headers are combined with SAFELISTED_HEADERS (which don't contain whitespace)
- **Line 67**: The problematic line that lowercases but doesn't strip: `self.allow_headers = [h.lower() for h in allow_headers]`
- **Line 129**: The validation that strips before checking: `if header.strip() not in self.allow_headers:`

According to HTTP specifications (RFC 7230), header field names are tokens that should not contain whitespace. While the CORS specification treats header names as case-insensitive, Starlette's validation logic already attempts to handle whitespace gracefully by stripping it, but the initialization doesn't match this behavior.

The Starlette documentation doesn't specify how whitespace in header configuration should be handled, leaving this behavior undefined and potentially surprising to users.

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