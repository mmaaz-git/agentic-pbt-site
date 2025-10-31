# Bug Report: starlette.middleware.wsgi SERVER_PORT Type Violation

**Target**: `starlette.middleware.wsgi.build_environ`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `build_environ` function in starlette's WSGI middleware sets the `SERVER_PORT` environment variable as an integer instead of a string, violating the WSGI specification (PEP 3333).

## Property-Based Test

```python
from hypothesis import given, settings
import hypothesis.strategies as st
from starlette.middleware.wsgi import build_environ


@given(st.integers(min_value=1, max_value=65535))
@settings(max_examples=100)
def test_server_port_type(port):
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "query_string": b"",
        "headers": [],
        "server": ("example.com", port),
        "http_version": "1.1",
    }

    environ = build_environ(scope, b"")

    assert isinstance(environ['SERVER_PORT'], str), f"SERVER_PORT should be str, got {type(environ['SERVER_PORT'])}"


if __name__ == "__main__":
    # Run the test
    test_server_port_type()
```

<details>

<summary>
**Failing input**: `port=1`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/29/hypo.py", line 26, in <module>
    test_server_port_type()
    ~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/29/hypo.py", line 7, in test_server_port_type
    @settings(max_examples=100)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/29/hypo.py", line 21, in test_server_port_type
    assert isinstance(environ['SERVER_PORT'], str), f"SERVER_PORT should be str, got {type(environ['SERVER_PORT'])}"
           ~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: SERVER_PORT should be str, got <class 'int'>
Falsifying example: test_server_port_type(
    port=1,  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
from starlette.middleware.wsgi import build_environ

scope = {
    "type": "http",
    "method": "GET",
    "path": "/",
    "query_string": b"",
    "headers": [],
    "server": ("example.com", 8000),
    "http_version": "1.1",
}

environ = build_environ(scope, b"")

print(f"SERVER_PORT value: {repr(environ['SERVER_PORT'])}")
print(f"SERVER_PORT type: {type(environ['SERVER_PORT'])}")
print(f"Is SERVER_PORT a string? {isinstance(environ['SERVER_PORT'], str)}")

# According to PEP 3333, SERVER_PORT must be a string
# Let's verify if it's actually an integer
assert isinstance(environ['SERVER_PORT'], str), f"SERVER_PORT should be str, got {type(environ['SERVER_PORT'])}"
```

<details>

<summary>
AssertionError: SERVER_PORT type mismatch
</summary>
```
SERVER_PORT value: 8000
SERVER_PORT type: <class 'int'>
Is SERVER_PORT a string? False
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/29/repo.py", line 21, in <module>
    assert isinstance(environ['SERVER_PORT'], str), f"SERVER_PORT should be str, got {type(environ['SERVER_PORT'])}"
           ~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: SERVER_PORT should be str, got <class 'int'>
```
</details>

## Why This Is A Bug

This violates the WSGI specification defined in PEP 3333, which explicitly requires `SERVER_PORT` to be a string. The specification states in the "environ Variables" section:

> `SERVER_PORT`: The port portion of the URL, as a **string**.

Additionally, this requirement originates from the CGI specification (RFC 3875, Section 4.1.15), which defines `SERVER_PORT` with the grammar: `SERVER_PORT = server-port` where `server-port = 1*digit`, indicating it must be a string containing one or more digit characters.

The current implementation directly assigns the integer port value from the server tuple without converting it to a string. This occurs on line 49 of `/home/npc/miniconda/lib/python3.13/site-packages/starlette/middleware/wsgi.py`:
```python
environ["SERVER_PORT"] = server[1]  # server[1] is an integer
```

This type mismatch can cause issues with WSGI applications that:
- Perform string operations on `SERVER_PORT` (e.g., concatenation, string formatting)
- Use strict type checking or validation
- Rely on the WSGI specification's guarantees about data types

## Relevant Context

- **Deprecation Status**: The WSGI middleware is deprecated as of the current version and will be removed in a future release. Users are directed to use https://github.com/abersheeran/a2wsgi as a replacement.
- **Code Location**: The bug is in `starlette/middleware/wsgi.py`, line 49, within the `build_environ` function.
- **WSGI Specification**: [PEP 3333](https://peps.python.org/pep-3333/#environ-variables) clearly defines the required types for environ variables.
- **CGI Specification**: [RFC 3875](https://datatracker.ietf.org/doc/html/rfc3875#section-4.1.15) defines the original CGI specification that WSGI inherits from.
- **Impact Scope**: Any WSGI application using Starlette's WSGI middleware that expects `SERVER_PORT` to be a string (as per spec) will receive an integer instead.

## Proposed Fix

```diff
--- a/starlette/middleware/wsgi.py
+++ b/starlette/middleware/wsgi.py
@@ -46,7 +46,7 @@ def build_environ(scope: Scope, body: bytes) -> dict[str, Any]:
     # Get server name and port - required in WSGI, not in ASGI
     server = scope.get("server") or ("localhost", 80)
     environ["SERVER_NAME"] = server[0]
-    environ["SERVER_PORT"] = server[1]
+    environ["SERVER_PORT"] = str(server[1])

     # Get client IP address
     if scope.get("client"):
```