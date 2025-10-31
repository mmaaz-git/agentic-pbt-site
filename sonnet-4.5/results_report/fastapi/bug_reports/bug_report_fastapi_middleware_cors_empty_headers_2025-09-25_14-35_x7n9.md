# Bug Report: CORSMiddleware Rejects Valid Requests with Empty Header Values

**Target**: `starlette.middleware.cors.CORSMiddleware.preflight_response` (re-exported as `fastapi.middleware.CORSMiddleware`)
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The CORSMiddleware incorrectly rejects preflight requests when the `Access-Control-Request-Headers` contains empty values from consecutive commas or trailing commas, even though such requests should be valid according to HTTP header parsing rules.

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
        allow_headers=valid_headers
    )

    requested_headers_str = ",".join(valid_headers) + ","

    request_headers = Headers({
        "origin": "http://example.com",
        "access-control-request-method": "POST",
        "access-control-request-headers": requested_headers_str
    })

    response = middleware.preflight_response(request_headers)

    assert response.status_code == 200, \
        f"Trailing comma should be ignored, but got {response.status_code}"
```

**Failing input**: `valid_headers=["content-type"]` with trailing comma

## Reproducing the Bug

```python
from starlette.middleware.cors import CORSMiddleware
from starlette.datastructures import Headers


class MockApp:
    pass


middleware = CORSMiddleware(
    MockApp(),
    allow_origins=["http://example.com"],
    allow_headers=["content-type", "x-custom"]
)

request_headers = Headers({
    "origin": "http://example.com",
    "access-control-request-method": "POST",
    "access-control-request-headers": "Content-Type,,X-Custom"
})

response = middleware.preflight_response(request_headers)

print(f"Status: {response.status_code}")
print(f"Body: {response.body}")
```

**Output**:
```
Status: 400
Body: b'Disallowed CORS headers'
```

Expected: Status 200 (empty values from consecutive commas should be ignored)

## Why This Is A Bug

According to HTTP header field parsing rules (RFC 7230 Section 7), empty list elements should be ignored when parsing comma-separated values. The current implementation at lines 127-131 of `starlette/middleware/cors.py`:

```python
elif requested_headers is not None:
    for header in [h.lower() for h in requested_headers.split(",")]:
        if header.strip() not in self.allow_headers:
            failures.append("headers")
            break
```

When splitting `"Content-Type,,X-Custom"` by comma, the result is `["Content-Type", "", "X-Custom"]`. The empty string element, even after `.strip()`, remains `""` and is checked against `self.allow_headers`. Since allowed headers typically don't include empty strings, the check fails and the request is rejected.

This can occur in practice when:
1. Headers are programmatically constructed with trailing commas
2. Header values contain extra commas due to improper formatting
3. Proxies or middleware modify headers in ways that introduce empty values

## Fix

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

This change:
1. Extracts the stripped value to avoid calling `strip()` twice
2. Skips empty strings (after stripping) entirely, treating them as non-existent per HTTP specifications
3. Only checks non-empty header values against the allowed list