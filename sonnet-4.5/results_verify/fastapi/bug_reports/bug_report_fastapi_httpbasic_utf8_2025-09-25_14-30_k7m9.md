# Bug Report: HTTPBasic Rejects Valid UTF-8 Credentials

**Target**: `fastapi.security.http.HTTPBasic`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

HTTPBasic authentication incorrectly decodes credentials as ASCII instead of UTF-8, causing it to reject valid credentials containing non-ASCII characters (e.g., accented letters, emoji, or international scripts), despite RFC 7617 requiring UTF-8 encoding.

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
```

**Failing input**: `username='0', password='\x80'`

## Reproducing the Bug

```python
import asyncio
from base64 import b64encode
from fastapi.security.http import HTTPBasic
from starlette.requests import Request


async def test_non_ascii_password():
    username = "user"
    password = "päss"

    credentials_str = f"{username}:{password}"
    encoded = b64encode(credentials_str.encode("utf-8")).decode("ascii")
    auth_header = f"Basic {encoded}"

    scope = {
        "type": "http",
        "method": "GET",
        "headers": [[b"authorization", auth_header.encode()]],
    }
    request = Request(scope)

    http_basic = HTTPBasic()
    result = await http_basic(request)

    print(f"Username: {result.username}")
    print(f"Password: {result.password}")


asyncio.run(test_non_ascii_password())
```

Running this code raises:
```
fastapi.exceptions.HTTPException: 401: Invalid authentication credentials
```

The credentials are valid per RFC 7617 but are rejected because the implementation uses ASCII decoding instead of UTF-8.

## Why This Is A Bug

RFC 7617 Section 2.1 explicitly states:

> "The user-id and password are transferred in a charset of UTF-8"

The current implementation at `fastapi/security/http.py:211` uses:
```python
data = b64decode(param).decode("ascii")
```

This violates the HTTP Basic Authentication specification. Valid credentials containing:
- Accented characters (café, José)
- Non-Latin scripts (用户, パスワード)
- Emoji or special symbols (🔑)

are incorrectly rejected as "Invalid authentication credentials".

## Fix

```diff
--- a/fastapi/security/http.py
+++ b/fastapi/security/http.py
@@ -208,7 +208,7 @@ class HTTPBasic(HTTPBase):
             headers=unauthorized_headers,
         )
     try:
-        data = b64decode(param).decode("ascii")
+        data = b64decode(param).decode("utf-8")
     except (ValueError, UnicodeDecodeError, binascii.Error):
         raise invalid_user_credentials_exc  # noqa: B904
     username, separator, password = data.partition(":")
```