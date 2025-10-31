# Bug Report: TrustedHostMiddleware IPv6 URL Parsing Crash

**Target**: `starlette.middleware.trustedhost.TrustedHostMiddleware`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

TrustedHostMiddleware crashes with `ValueError: Invalid IPv6 URL` when processing malformed host headers that resemble incomplete IPv6 addresses during www redirect operations, instead of returning the expected 400 Bad Request response.

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
    try:
        middleware = TrustedHostMiddleware(dummy_app, allowed_hosts=[pattern], www_redirect=True)
    except AssertionError:
        # Skip hosts that violate the wildcard pattern rules
        return

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


if __name__ == "__main__":
    test_trustedhost_www_redirect_should_not_crash()
```

<details>

<summary>
**Failing input**: `host='['`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/57/hypo.py", line 55, in <module>
    test_trustedhost_www_redirect_should_not_crash()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/57/hypo.py", line 7, in test_trustedhost_www_redirect_should_not_crash
    @settings(max_examples=200)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/57/hypo.py", line 51, in test_trustedhost_www_redirect_should_not_crash
    asyncio.run(run_test())
    ~~~~~~~~~~~^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/asyncio/runners.py", line 195, in run
    return runner.run(main)
           ~~~~~~~~~~^^^^^^
  File "/home/npc/miniconda/lib/python3.13/asyncio/runners.py", line 118, in run
    return self._loop.run_until_complete(task)
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^
  File "/home/npc/miniconda/lib/python3.13/asyncio/base_events.py", line 725, in run_until_complete
    return future.result()
           ~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/57/hypo.py", line 48, in run_test
    await middleware(scope, receive, send)
  File "/home/npc/miniconda/lib/python3.13/site-packages/starlette/middleware/trustedhost.py", line 56, in __call__
    redirect_url = url.replace(netloc="www." + url.netloc)
                                               ^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/starlette/datastructures.py", line 76, in netloc
    return self.components.netloc
           ^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/starlette/datastructures.py", line 67, in components
    self._components = urlsplit(self._url)
                       ~~~~~~~~^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/urllib/parse.py", line 514, in urlsplit
    raise ValueError("Invalid IPv6 URL")
ValueError: Invalid IPv6 URL
Falsifying example: test_trustedhost_www_redirect_should_not_crash(
    host='[',
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/miniconda/lib/python3.13/asyncio/runners.py:119
        /home/npc/miniconda/lib/python3.13/urllib/parse.py:514
```
</details>

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


if __name__ == "__main__":
    asyncio.run(test_crash())
```

<details>

<summary>
ValueError: Invalid IPv6 URL
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/57/repo.py", line 41, in <module>
    asyncio.run(test_crash())
    ~~~~~~~~~~~^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/asyncio/runners.py", line 195, in run
    return runner.run(main)
           ~~~~~~~~~~^^^^^^
  File "/home/npc/miniconda/lib/python3.13/asyncio/runners.py", line 118, in run
    return self._loop.run_until_complete(task)
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^
  File "/home/npc/miniconda/lib/python3.13/asyncio/base_events.py", line 725, in run_until_complete
    return future.result()
           ~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/57/repo.py", line 37, in test_crash
    await middleware(scope, receive, send)
  File "/home/npc/miniconda/lib/python3.13/site-packages/starlette/middleware/trustedhost.py", line 56, in __call__
    redirect_url = url.replace(netloc="www." + url.netloc)
                                               ^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/starlette/datastructures.py", line 76, in netloc
    return self.components.netloc
           ^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/starlette/datastructures.py", line 67, in components
    self._components = urlsplit(self._url)
                       ~~~~~~~~^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/urllib/parse.py", line 514, in urlsplit
    raise ValueError("Invalid IPv6 URL")
ValueError: Invalid IPv6 URL
```
</details>

## Why This Is A Bug

The TrustedHostMiddleware is designed to protect against host header attacks by validating incoming host headers against a list of allowed hosts. According to its implementation pattern, when an invalid host is detected, the middleware should return a 400 Bad Request response with the message "Invalid host header".

However, when the middleware attempts to perform a www redirect (when `www_redirect=True` and the host without "www." prefix matches an allowed pattern), it constructs a URL object from the ASGI scope to build the redirect URL. If the host header contains characters that Python's `urllib.parse.urlsplit()` interprets as an incomplete IPv6 address (such as a lone `[` bracket), the URL parsing fails with `ValueError: Invalid IPv6 URL`.

This crash violates the expected behavior in several ways:

1. **Security middleware should never crash on malformed input** - As the first line of defense against host header attacks, TrustedHostMiddleware should gracefully handle any form of malformed input
2. **Inconsistent error handling** - The middleware already returns 400 responses for other invalid host scenarios, but crashes in this specific case
3. **Potential DoS vector** - An attacker could send requests with malformed host headers like `[` to trigger application crashes

The root cause is in line 55-57 of `trustedhost.py` where the URL construction is not wrapped in error handling:
```python
url = URL(scope=scope)
redirect_url = url.replace(netloc="www." + url.netloc)
response = RedirectResponse(url=str(redirect_url))
```

## Relevant Context

The crash occurs specifically when:
1. The middleware is configured with `www_redirect=True`
2. An allowed host pattern like `"www.["` is configured
3. A request arrives with host header `"["` (without the www prefix)
4. The middleware attempts to redirect to `"www.["` but crashes during URL construction

The URL class (from `starlette.datastructures`) constructs a URL string from the scope and then uses Python's `urllib.parse.urlsplit()` to parse it. When the host is `[`, it creates `http://[/path` which `urlsplit()` interprets as an invalid IPv6 address format (IPv6 addresses in URLs must be enclosed in brackets like `[::1]`, but a lone `[` is invalid).

Related code locations:
- `starlette/middleware/trustedhost.py:55-57` - Where the crash occurs during redirect
- `starlette/datastructures.py:67` - Where urlsplit is called on the constructed URL
- `urllib/parse.py:514` - Where the "Invalid IPv6 URL" error is raised

## Proposed Fix

```diff
--- a/starlette/middleware/trustedhost.py
+++ b/starlette/middleware/trustedhost.py
@@ -52,9 +52,14 @@ class TrustedHostMiddleware:
             await self.app(scope, receive, send)
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