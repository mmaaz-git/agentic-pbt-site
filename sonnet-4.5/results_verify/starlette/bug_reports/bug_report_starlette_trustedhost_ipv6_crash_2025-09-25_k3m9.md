# Bug Report: TrustedHostMiddleware IPv6 URL Parsing Crash

**Target**: `starlette.middleware.trustedhost.TrustedHostMiddleware`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

TrustedHostMiddleware crashes with `ValueError: Invalid IPv6 URL` when processing certain malformed host headers during www redirect, instead of returning a 400 Bad Request response.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import asyncio
from starlette.middleware.trustedhost import TrustedHostMiddleware


@given(st.text(min_size=1, max_size=20))
@settings(max_examples=200)
def test_trustedhost_www_redirect_should_not_crash(host):
    async def dummy_app(scope, receive, send):
        await send({
            "type": "http.response.start",
            "status": 200,
            "headers": [[b"content-type", b"text/plain"]],
        })
        await send({
            "type": "http.response.body",
            "body": b"OK",
        })

    pattern = f"www.{host}"
    middleware = TrustedHostMiddleware(dummy_app, allowed_hosts=[pattern], www_redirect=True)

    scope = {
        "type": "http",
        "method": "GET",
        "scheme": "http",
        "path": "/test",
        "query_string": b"",
        "server": (host, 80),
        "headers": [
            [b"host", host.encode()],
        ],
    }

    received_messages = []

    async def receive():
        return {"type": "http.request", "body": b""}

    async def send(message):
        received_messages.append(message)

    async def run_test():
        await middleware(scope, receive, send)
        assert len(received_messages) >= 2

    asyncio.run(run_test())
```

**Failing input**: `host='['`

## Reproducing the Bug

```python
import asyncio
from starlette.middleware.trustedhost import TrustedHostMiddleware


async def dummy_app(scope, receive, send):
    await send({
        "type": "http.response.start",
        "status": 200,
        "headers": [[b"content-type", b"text/plain"]],
    })
    await send({"type": "http.response.body", "body": b"OK"})


async def test_crash():
    middleware = TrustedHostMiddleware(
        dummy_app,
        allowed_hosts=["www.["],
        www_redirect=True
    )

    scope = {
        "type": "http",
        "method": "GET",
        "scheme": "http",
        "path": "/",
        "query_string": b"",
        "server": ("[", 80),
        "headers": [[b"host", b"["]],
    }

    async def receive():
        return {"type": "http.request", "body": b""}

    async def send(message):
        pass

    await middleware(scope, receive, send)


asyncio.run(test_crash())
```

## Why This Is A Bug

When TrustedHostMiddleware attempts to perform a www redirect, it constructs a `URL` object from the ASGI scope. If the host header contains invalid characters that trigger URL parsing errors (like a lone `[` which looks like an incomplete IPv6 address), the middleware crashes with `ValueError: Invalid IPv6 URL` instead of gracefully returning a 400 Bad Request response.

This is problematic because:
1. Middleware should handle malformed input gracefully, not crash
2. The middleware already validates hosts and returns 400 for invalid hosts in other cases
3. Malformed host headers from clients should not crash the server

## Fix

```diff
--- a/starlette/middleware/trustedhost.py
+++ b/starlette/middleware/trustedhost.py
@@ -53,8 +53,13 @@ class TrustedHostMiddleware:
         else:
             response: Response
             if found_www_redirect and self.www_redirect:
-                url = URL(scope=scope)
-                redirect_url = url.replace(netloc="www." + url.netloc)
-                response = RedirectResponse(url=str(redirect_url))
+                try:
+                    url = URL(scope=scope)
+                    redirect_url = url.replace(netloc="www." + url.netloc)
+                    response = RedirectResponse(url=str(redirect_url))
+                except ValueError:
+                    # Invalid URL in scope, treat as invalid host
+                    response = PlainTextResponse("Invalid host header", status_code=400)
             else:
                 response = PlainTextResponse("Invalid host header", status_code=400)
             await response(scope, receive, send)
```