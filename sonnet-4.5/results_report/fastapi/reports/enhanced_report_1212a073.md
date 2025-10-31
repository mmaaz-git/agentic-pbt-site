# Bug Report: fastapi.middleware.cors.CORSMiddleware Rejects Valid CORS Requests with Empty Header Values

**Target**: `starlette.middleware.cors.CORSMiddleware.preflight_response` (re-exported as `fastapi.middleware.CORSMiddleware`)
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

CORSMiddleware incorrectly rejects valid CORS preflight requests when the `Access-Control-Request-Headers` header contains empty values resulting from consecutive commas, trailing commas, or leading commas in the comma-separated list.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from starlette.middleware.cors import CORSMiddleware
from starlette.datastructures import Headers
from starlette.responses import Response


@given(
    st.lists(st.sampled_from(["content-type", "x-custom", "authorization"]), min_size=1, max_size=5)
)
def test_cors_empty_header_values_ignored(valid_headers):
    class MockApp:
        async def __call__(self, scope, receive, send):
            pass

    middleware = CORSMiddleware(
        MockApp(),
        allow_origins=["http://example.com"],
        allow_methods=["GET", "POST"],
        allow_headers=valid_headers
    )

    # Test with trailing comma
    requested_headers_str = ",".join(valid_headers) + ","

    request_headers = Headers({
        "origin": "http://example.com",
        "access-control-request-method": "POST",
        "access-control-request-headers": requested_headers_str
    })

    response = middleware.preflight_response(request_headers)

    assert response.status_code == 200, \
        f"Trailing comma should be ignored, but got {response.status_code} for headers: {requested_headers_str}"


if __name__ == "__main__":
    # Run the test to find a failing example
    test_cors_empty_header_values_ignored()
```

<details>

<summary>
**Failing input**: `valid_headers=['content-type']`
</summary>
```
Traceback (most recent call last):
  File "<string>", line 1, in <module>
    from hypo import test_cors_empty_header_values_ignored; test_cors_empty_header_values_ignored()
                                                            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/20/hypo.py", line 8, in test_cors_empty_header_values_ignored
    st.lists(st.sampled_from(["content-type", "x-custom", "authorization"]), min_size=1, max_size=5)
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/20/hypo.py", line 33, in test_cors_empty_header_values_ignored
    assert response.status_code == 200, \
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Trailing comma should be ignored, but got 400 for headers: content-type,
Falsifying example: test_cors_empty_header_values_ignored(
    valid_headers=['content-type'],
)
```
</details>

## Reproducing the Bug

```python
from starlette.middleware.cors import CORSMiddleware
from starlette.datastructures import Headers


class MockApp:
    pass


middleware = CORSMiddleware(
    MockApp(),
    allow_origins=["http://example.com"],
    allow_methods=["GET", "POST"],
    allow_headers=["content-type", "x-custom"]
)

# Test case with consecutive commas (empty value in the middle)
request_headers = Headers({
    "origin": "http://example.com",
    "access-control-request-method": "POST",
    "access-control-request-headers": "Content-Type,,X-Custom"
})

response = middleware.preflight_response(request_headers)

print(f"Test 1: Consecutive commas")
print(f"Headers: 'Content-Type,,X-Custom'")
print(f"Status: {response.status_code}")
print(f"Body: {response.body}")
print()

# Test case with trailing comma
request_headers2 = Headers({
    "origin": "http://example.com",
    "access-control-request-method": "POST",
    "access-control-request-headers": "Content-Type,X-Custom,"
})

response2 = middleware.preflight_response(request_headers2)

print(f"Test 2: Trailing comma")
print(f"Headers: 'Content-Type,X-Custom,'")
print(f"Status: {response2.status_code}")
print(f"Body: {response2.body}")
print()

# Test case with leading comma
request_headers3 = Headers({
    "origin": "http://example.com",
    "access-control-request-method": "POST",
    "access-control-request-headers": ",Content-Type,X-Custom"
})

response3 = middleware.preflight_response(request_headers3)

print(f"Test 3: Leading comma")
print(f"Headers: ',Content-Type,X-Custom'")
print(f"Status: {response3.status_code}")
print(f"Body: {response3.body}")
print()

# Test case without extra commas (should work)
request_headers4 = Headers({
    "origin": "http://example.com",
    "access-control-request-method": "POST",
    "access-control-request-headers": "Content-Type,X-Custom"
})

response4 = middleware.preflight_response(request_headers4)

print(f"Test 4: Normal case (no extra commas)")
print(f"Headers: 'Content-Type,X-Custom'")
print(f"Status: {response4.status_code}")
print(f"Body: {response4.body}")
```

<details>

<summary>
All tests with extra commas fail with HTTP 400
</summary>
```
Test 1: Consecutive commas
Headers: 'Content-Type,,X-Custom'
Status: 400
Body: b'Disallowed CORS headers'

Test 2: Trailing comma
Headers: 'Content-Type,X-Custom,'
Status: 400
Body: b'Disallowed CORS headers'

Test 3: Leading comma
Headers: ',Content-Type,X-Custom'
Status: 400
Body: b'Disallowed CORS headers'

Test 4: Normal case (no extra commas)
Headers: 'Content-Type,X-Custom'
Status: 200
Body: b'OK'
```
</details>

## Why This Is A Bug

The bug violates the principle of robustness in network protocol implementations ("be conservative in what you send, liberal in what you accept" - Postel's Law). When the middleware splits a header value like `"Content-Type,,"` by comma, it produces the array `["Content-Type", "", ""]`. The empty strings are then checked against the `allow_headers` list, which naturally doesn't contain empty string entries, causing the validation to fail.

Empty values in comma-separated lists carry no semantic meaning and should be ignored. While HTTP specifications (RFC 7230) don't explicitly mandate ignoring empty values in comma-separated header lists, common practice in HTTP parsing is to treat them as non-existent. The W3C CORS specification doesn't provide explicit guidance on this edge case either.

The bug occurs at lines 127-131 of `starlette/middleware/cors.py`:

```python
elif requested_headers is not None:
    for header in [h.lower() for h in requested_headers.split(",")]:
        if header.strip() not in self.allow_headers:
            failures.append("headers")
            break
```

The code correctly strips whitespace but still validates empty strings against the allowed headers list, causing legitimate requests to be rejected.

## Relevant Context

This issue commonly occurs in production environments when:

1. **Programmatically generated headers**: Client libraries or frameworks may add trailing commas when building comma-separated header lists
2. **Proxy modifications**: Intermediate proxies or load balancers may modify headers in ways that introduce extra commas
3. **JavaScript string concatenation**: Frontend code building header strings dynamically may inadvertently include extra commas
4. **Copy-paste errors**: Developers copying header configurations may accidentally include formatting issues

The FastAPI middleware is a direct re-export of Starlette's implementation (`fastapi/middleware/cors.py` line 1):
```python
from starlette.middleware.cors import CORSMiddleware as CORSMiddleware
```

Therefore, this bug affects both FastAPI and Starlette users. The issue is particularly problematic because CORS errors are often difficult to debug, and the error message "Disallowed CORS headers" doesn't indicate that the problem is an empty value rather than an actually disallowed header.

## Proposed Fix

```diff
--- a/starlette/middleware/cors.py
+++ b/starlette/middleware/cors.py
@@ -126,7 +126,8 @@ class CORSMiddleware:
             headers["Access-Control-Allow-Headers"] = requested_headers
         elif requested_headers is not None:
             for header in [h.lower() for h in requested_headers.split(",")]:
-                if header.strip() not in self.allow_headers:
+                stripped = header.strip()
+                if stripped and stripped not in self.allow_headers:
                     failures.append("headers")
                     break
```