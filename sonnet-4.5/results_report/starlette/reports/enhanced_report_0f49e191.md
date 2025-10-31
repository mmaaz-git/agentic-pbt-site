# Bug Report: starlette.middleware.wsgi.build_environ SERVER_PORT Type Violation

**Target**: `starlette.middleware.wsgi.build_environ`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `build_environ` function in Starlette's WSGI middleware returns SERVER_PORT as an integer instead of a string, violating the WSGI specification (PEP 3333) which explicitly requires it to be a string type.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from starlette.middleware.wsgi import build_environ


@given(port=st.integers(min_value=1, max_value=65535))
@settings(max_examples=200)
def test_server_port_is_string_per_pep3333(port):
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

    assert isinstance(environ["SERVER_PORT"], str), \
        f"PEP 3333 requires SERVER_PORT to be a string, got {type(environ['SERVER_PORT'])}"

if __name__ == "__main__":
    test_server_port_is_string_per_pep3333()
```

<details>

<summary>
**Failing input**: `port=1` (or any integer from 1-65535)
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/33/hypo.py", line 24, in <module>
    test_server_port_is_string_per_pep3333()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/33/hypo.py", line 6, in test_server_port_is_string_per_pep3333
    @settings(max_examples=200)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/33/hypo.py", line 20, in test_server_port_is_string_per_pep3333
    assert isinstance(environ["SERVER_PORT"], str), \
           ~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: PEP 3333 requires SERVER_PORT to be a string, got <class 'int'>
Falsifying example: test_server_port_is_string_per_pep3333(
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
    "path": "/test",
    "query_string": b"",
    "headers": [],
    "server": ("example.com", 8080),
    "http_version": "1.1",
}

environ = build_environ(scope, b"")
print(f"SERVER_PORT: {environ['SERVER_PORT']!r}")
print(f"Type: {type(environ['SERVER_PORT'])}")
print(f"Is string: {isinstance(environ['SERVER_PORT'], str)}")
print(f"PEP 3333 requires SERVER_PORT to be a string, but got {type(environ['SERVER_PORT']).__name__}")
```

<details>

<summary>
SERVER_PORT is returned as int instead of str
</summary>
```
SERVER_PORT: 8080
Type: <class 'int'>
Is string: False
PEP 3333 requires SERVER_PORT to be a string, but got int
```
</details>

## Why This Is A Bug

This violates the WSGI specification defined in PEP 3333, which states unambiguously:

> **SERVER_PORT**: The port portion of the server name, as a string.

The specification further clarifies that "SERVER_NAME and SERVER_PORT are required strings and must never be empty." The WSGI standard follows the CGI (Common Gateway Interface) specification where all environment variables are strings by definition.

While many WSGI applications may handle integer SERVER_PORT values gracefully through implicit type conversion, this deviation from the specification can cause failures in:
- Applications that perform string operations on SERVER_PORT (e.g., concatenation, formatting)
- Strict WSGI compliance validators
- Applications that compare SERVER_PORT with string values
- Middleware that expects specification-compliant environ dictionaries

The module includes a deprecation warning directing users to a2wsgi as a replacement, but as long as the module remains in the codebase, it should conform to the WSGI specification it claims to implement.

## Relevant Context

The bug occurs in `/home/npc/pbt/agentic-pbt/envs/starlette_env/lib/python3.13/site-packages/starlette/middleware/wsgi.py` at line 50 within the `build_environ` function. The function is responsible for converting ASGI scope objects into WSGI environ dictionaries.

PEP 3333 documentation: https://peps.python.org/pep-3333/#environ-variables

The function correctly handles other required string conversions (e.g., SERVER_NAME at line 49), but misses the conversion for SERVER_PORT.

## Proposed Fix

```diff
--- a/starlette/middleware/wsgi.py
+++ b/starlette/middleware/wsgi.py
@@ -47,7 +47,7 @@ def build_environ(scope: Scope, body: bytes) -> dict[str, Any]:
     # Get server name and port - required in WSGI, not in ASGI
     server = scope.get("server") or ("localhost", 80)
     environ["SERVER_NAME"] = server[0]
-    environ["SERVER_PORT"] = server[1]
+    environ["SERVER_PORT"] = str(server[1])

     # Get client IP address
     if scope.get("client"):
```