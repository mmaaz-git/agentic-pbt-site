# Bug Report: starlette.middleware.cors.CORSMiddleware Whitespace Handling in allow_origins

**Target**: `starlette.middleware.cors.CORSMiddleware`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

CORSMiddleware incorrectly rejects valid CORS preflight requests when origin URLs in the `allow_origins` configuration contain leading or trailing whitespace, because it performs exact string matching without normalization.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from starlette.datastructures import Headers
from starlette.middleware.cors import CORSMiddleware


@given(
    origin=st.text(min_size=1, max_size=30, alphabet="abcdefghijklmnopqrstuvwxyz:/.-"),
    spaces_before=st.integers(min_value=0, max_value=3),
    spaces_after=st.integers(min_value=0, max_value=3)
)
def test_cors_allow_origins_whitespace(origin, spaces_before, spaces_after):
    origin_with_spaces = " " * spaces_before + origin + " " * spaces_after

    middleware = CORSMiddleware(
        app=None,
        allow_origins=[origin_with_spaces]
    )

    request_headers = Headers({
        "origin": origin,
        "access-control-request-method": "GET"
    })

    response = middleware.preflight_response(request_headers)

    assert response.status_code == 200, \
        f"Expected 200 OK but got {response.status_code}. " \
        f"Origin '{origin_with_spaces}' (with spaces) was allowed in config, " \
        f"but request origin '{origin}' (without spaces) was rejected."

if __name__ == "__main__":
    test_cors_allow_origins_whitespace()
```

<details>

<summary>
**Failing input**: `origin=':'`, `spaces_before=0`, `spaces_after=1`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/22/hypo.py", line 32, in <module>
    test_cors_allow_origins_whitespace()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/22/hypo.py", line 7, in test_cors_allow_origins_whitespace
    origin=st.text(min_size=1, max_size=30, alphabet="abcdefghijklmnopqrstuvwxyz:/.-"),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/22/hypo.py", line 26, in test_cors_allow_origins_whitespace
    assert response.status_code == 200, \
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Expected 200 OK but got 400. Origin ': ' (with spaces) was allowed in config, but request origin ':' (without spaces) was rejected.
Falsifying example: test_cors_allow_origins_whitespace(
    origin=':',
    spaces_before=0,
    spaces_after=1,
)
```
</details>

## Reproducing the Bug

```python
from starlette.datastructures import Headers
from starlette.middleware.cors import CORSMiddleware

# Demonstrate the bug with a trailing space in the allowed origin
middleware = CORSMiddleware(
    app=None,
    allow_origins=["http://example.com "]  # Note: trailing space
)

request_headers = Headers({
    "origin": "http://example.com",  # No trailing space
    "access-control-request-method": "GET"
})

response = middleware.preflight_response(request_headers)

print(f"Configuration: allow_origins=['http://example.com '] (with trailing space)")
print(f"Request origin: 'http://example.com' (no space)")
print(f"Response status: {response.status_code}")
print(f"Expected: 200 (origin should be allowed)")
print(f"Actual: {response.status_code}")

if response.status_code != 200:
    print(f"\nERROR: Valid request was rejected!")
    print(f"Response body: {response.body.decode()}")
```

<details>

<summary>
CORS preflight request rejected with status 400
</summary>
```
Configuration: allow_origins=['http://example.com '] (with trailing space)
Request origin: 'http://example.com' (no space)
Response status: 400
Expected: 200 (origin should be allowed)
Actual: 400

ERROR: Valid request was rejected!
Response body: Disallowed CORS origin
```
</details>

## Why This Is A Bug

This violates expected behavior because the middleware inconsistently handles normalization between different configuration parameters. The code at `/home/npc/pbt/agentic-pbt/envs/starlette_env/lib/python3.13/site-packages/starlette/middleware/cors.py` shows:

1. **Headers ARE normalized** (line 67): `self.allow_headers = [h.lower() for h in allow_headers]`
2. **Request headers ARE stripped** (line 129): `if header.strip() not in self.allow_headers`
3. **Origins are NOT normalized** (line 65): `self.allow_origins = allow_origins`
4. **Exact matching fails** (line 102): `return origin in self.allow_origins`

This inconsistency means that while header matching is forgiving of whitespace, origin matching is not. Since browsers send origin headers without extraneous whitespace (per RFC 6454), any whitespace in the configuration will cause valid requests to be rejected with a 400 error.

## Relevant Context

- **RFC 6454** defines the Origin header serialization format without leading/trailing whitespace
- **Common sources of whitespace**: Configuration files (YAML/JSON formatting), environment variables, copy-paste from documentation
- **Starlette documentation** does not warn about whitespace sensitivity for origins
- **Similar parameters** like `allow_headers` already perform normalization, setting user expectation
- **Error message** ("Disallowed CORS origin") doesn't indicate the whitespace issue, making debugging difficult

Code location: `/home/npc/pbt/agentic-pbt/envs/starlette_env/lib/python3.13/site-packages/starlette/middleware/cors.py:65`

## Proposed Fix

```diff
--- a/starlette/middleware/cors.py
+++ b/starlette/middleware/cors.py
@@ -62,7 +62,7 @@ class CORSMiddleware:
             preflight_headers["Access-Control-Allow-Credentials"] = "true"

         self.app = app
-        self.allow_origins = allow_origins
+        self.allow_origins = [origin.strip() for origin in allow_origins]
         self.allow_methods = allow_methods
         self.allow_headers = [h.lower() for h in allow_headers]
         self.allow_all_origins = allow_all_origins
```