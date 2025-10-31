# Bug Report: GZipMiddleware Invalid compresslevel Causes ValueError

**Target**: `fastapi.middleware.gzip.GZipMiddleware`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `GZipMiddleware` accepts any integer value for `compresslevel` parameter without validation. When a value outside the valid range (-1 to 9) is provided, the middleware crashes with `ValueError: Invalid initialization option` when processing a request with gzip encoding.

## Property-Based Test

```python
from hypothesis import given, strategies as st, example
from fastapi.middleware.gzip import GZipMiddleware
import pytest


class DummyApp:
    pass


@given(st.integers(min_value=10, max_value=100))
@example(10)
def test_gzipmiddleware_invalid_compresslevel_crashes(compresslevel):
    middleware = GZipMiddleware(DummyApp(), compresslevel=compresslevel)

    import asyncio
    from starlette.datastructures import Headers

    scope = {
        "type": "http",
        "method": "GET",
        "headers": [(b"accept-encoding", b"gzip")],
    }

    async def dummy_receive():
        return {"type": "http.request", "body": b""}

    async def dummy_send(message):
        pass

    async def test():
        await middleware(scope, dummy_receive, dummy_send)

    with pytest.raises(ValueError, match="Invalid initialization option"):
        asyncio.run(test())
```

**Failing input**: `compresslevel=10`

## Reproducing the Bug

```python
import asyncio
from fastapi.middleware.gzip import GZipMiddleware


class DummyApp:
    async def __call__(self, scope, receive, send):
        await send({
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-type", b"text/plain")],
        })
        await send({
            "type": "http.response.body",
            "body": b"Hello, World!",
        })


middleware = GZipMiddleware(DummyApp(), compresslevel=10)

scope = {
    "type": "http",
    "method": "GET",
    "headers": [(b"accept-encoding", b"gzip")],
}


async def receive():
    return {"type": "http.request", "body": b""}


messages = []


async def send(message):
    messages.append(message)


async def test():
    await middleware(scope, receive, send)


asyncio.run(test())
```

## Why This Is A Bug

The `compresslevel` parameter for Python's `gzip.GzipFile` must be between -1 (default) and 9 inclusive. Values outside this range cause `ValueError` to be raised. The `GZipMiddleware` should validate this parameter at initialization time to provide a clear error message, rather than allowing the middleware to be created and then crashing when processing requests.

This violates the principle of fail-fast: the middleware silently accepts invalid configuration and only fails when it's actually used, making debugging difficult.

## Fix

```diff
--- a/starlette/middleware/gzip.py
+++ b/starlette/middleware/gzip.py
@@ -18,6 +18,8 @@ class GZipMiddleware:
     def __init__(self, app: ASGIApp, minimum_size: int = 500, compresslevel: int = 9) -> None:
+        if not (-1 <= compresslevel <= 9):
+            raise ValueError(f"compresslevel must be between -1 and 9, got {compresslevel}")
         self.app = app
         self.minimum_size = minimum_size
         self.compresslevel = compresslevel
```