# Bug Report: fastapi.security.http.HTTPBasic Rejects Valid UTF-8 Credentials

**Target**: `fastapi.security.http.HTTPBasic`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

HTTPBasic authentication in FastAPI incorrectly decodes base64-encoded credentials using ASCII instead of UTF-8, causing valid credentials containing non-ASCII characters to be rejected with a 401 error.

## Property-Based Test

```python
import asyncio
from base64 import b64encode
from hypothesis import given, strategies as st, settings, assume
from fastapi.security.http import HTTPBasic, HTTPBasicCredentials
from starlette.requests import Request


async def create_test_request(authorization_header):
    scope = {
        "type": "http",
        "method": "GET",
        "headers": [[b"authorization", authorization_header.encode()]],
    }
    return Request(scope)


@given(
    st.text(min_size=1).filter(lambda s: ':' not in s and '\x00' not in s),
    st.text().filter(lambda s: '\x00' not in s)
)
@settings(max_examples=200)
def test_http_basic_auth_roundtrip(username, password):
    credentials_str = f"{username}:{password}"

    try:
        encoded = b64encode(credentials_str.encode("utf-8")).decode("ascii")
    except:
        assume(False)

    auth_header = f"Basic {encoded}"

    async def run_test():
        request = await create_test_request(auth_header)
        http_basic = HTTPBasic()
        result = await http_basic(request)

        assert result is not None
        assert isinstance(result, HTTPBasicCredentials)
        assert result.username == username
        assert result.password == password

    asyncio.run(run_test())


if __name__ == "__main__":
    test_http_basic_auth_roundtrip()
```

<details>

<summary>
**Failing input**: `username='0', password='\x80'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/miniconda/lib/python3.13/site-packages/fastapi/security/http.py", line 211, in __call__
    data = b64decode(param).decode("ascii")
UnicodeDecodeError: 'ascii' codec can't decode byte 0xc2 in position 2: ordinal not in range(128)

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/5/hypo.py", line 46, in <module>
    test_http_basic_auth_roundtrip()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/5/hypo.py", line 18, in test_http_basic_auth_roundtrip
    st.text(min_size=1).filter(lambda s: ':' not in s and '\x00' not in s),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/5/hypo.py", line 42, in test_http_basic_auth_roundtrip
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
  File "/home/npc/pbt/agentic-pbt/worker_/5/hypo.py", line 35, in run_test
    result = await http_basic(request)
             ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/fastapi/security/http.py", line 213, in __call__
    raise invalid_user_credentials_exc  # noqa: B904
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
fastapi.exceptions.HTTPException: 401: Invalid authentication credentials
Falsifying example: test_http_basic_auth_roundtrip(
    username='0',  # or any other generated value
    password='\x80',
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/miniconda/lib/python3.13/site-packages/fastapi/security/http.py:212
        /home/npc/miniconda/lib/python3.13/asyncio/runners.py:119
```
</details>

## Reproducing the Bug

```python
import asyncio
from base64 import b64encode
from fastapi.security.http import HTTPBasic
from starlette.requests import Request


async def test_utf8_credentials():
    # Test case with non-ASCII character
    username = "0"
    password = "\x80"  # Non-ASCII byte (128 in decimal)

    # Encode credentials as per HTTP Basic Auth spec
    credentials_str = f"{username}:{password}"
    encoded = b64encode(credentials_str.encode("utf-8")).decode("ascii")
    auth_header = f"Basic {encoded}"

    # Create a mock request with the authorization header
    scope = {
        "type": "http",
        "method": "GET",
        "headers": [[b"authorization", auth_header.encode()]],
    }
    request = Request(scope)

    # Try to authenticate with HTTPBasic
    http_basic = HTTPBasic()
    try:
        result = await http_basic(request)
        print(f"Success! Username: {result.username}, Password: {result.password}")
    except Exception as e:
        print(f"Error: {e}")
        print(f"Error type: {type(e).__name__}")


if __name__ == "__main__":
    asyncio.run(test_utf8_credentials())
```

<details>

<summary>
HTTPException raised with 401 status code
</summary>
```
Error: 401: Invalid authentication credentials
Error type: HTTPException
```
</details>

## Why This Is A Bug

This violates the expected behavior of HTTP Basic Authentication in several ways:

1. **RFC 7617 Compliance**: While RFC 7617 doesn't mandate UTF-8 as the absolute default encoding, it clearly expects implementations to support Unicode characters. Section 2.1 states that implementations must support the "UsernameCasePreserved" profile for usernames and "OpaqueString" profile for passwords, both of which include Unicode characters beyond ASCII.

2. **Modern Web Standards**: In 2025, web applications are expected to support international users. Restricting authentication to ASCII-only characters excludes users who need to use:
   - Accented characters (café, José, André)
   - Non-Latin scripts (中文, 日本語, العربية, русский)
   - Emoji or special symbols that users might include in passwords

3. **Inconsistent with Base64 Encoding**: The credentials are properly base64-encoded in UTF-8 by the client (which is correct), but the server decodes them as ASCII. This creates an asymmetry where valid UTF-8 strings that successfully encode to base64 fail to authenticate.

4. **Silent Failure**: The error message "Invalid authentication credentials" doesn't indicate that the issue is with character encoding, making it difficult for developers to diagnose why valid credentials are being rejected.

## Relevant Context

The bug occurs in `/home/npc/pbt/agentic-pbt/envs/fastapi_env/lib/python3.13/site-packages/fastapi/security/http.py` at line 211:

```python
try:
    data = b64decode(param).decode("ascii")  # <-- This is the problematic line
except (ValueError, UnicodeDecodeError, binascii.Error):
    raise invalid_user_credentials_exc
```

When the base64-decoded bytes contain non-ASCII characters (any byte value > 127), the `decode("ascii")` call raises a `UnicodeDecodeError`, which is caught and converted to an HTTP 401 error.

Key observations:
- The fix is backwards-compatible since UTF-8 is a superset of ASCII
- Other major HTTP Basic authentication implementations (e.g., in Flask, Django) typically support UTF-8
- FastAPI's documentation doesn't mention this ASCII-only limitation, so users would reasonably expect UTF-8 support

RFC 7617 reference: https://datatracker.ietf.org/doc/html/rfc7617

## Proposed Fix

```diff
--- a/fastapi/security/http.py
+++ b/fastapi/security/http.py
@@ -208,7 +208,7 @@ class HTTPBasic(HTTPBase):
             headers=unauthorized_headers,
         )
         try:
-            data = b64decode(param).decode("ascii")
+            data = b64decode(param).decode("utf-8")
         except (ValueError, UnicodeDecodeError, binascii.Error):
             raise invalid_user_credentials_exc  # noqa: B904
         username, separator, password = data.partition(":")
```