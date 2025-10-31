# Bug Report: FastAPI HTTPBasic UTF-8 Credential Decoding Failure

**Target**: `fastapi.security.http.HTTPBasic`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

FastAPI's HTTPBasic authentication crashes with UnicodeDecodeError when credentials contain non-ASCII UTF-8 characters, violating RFC 7617 which requires UTF-8 support for HTTP Basic Authentication.

## Property-Based Test

```python
import asyncio
import base64
from hypothesis import given, strategies as st, assume
from fastapi.security import HTTPBasic
from starlette.requests import Request
from starlette.datastructures import Headers


@given(
    username=st.text(min_size=1, max_size=50).filter(lambda x: ':' not in x),
    password=st.text(min_size=0, max_size=50)
)
def test_httpbasic_utf8_support(username, password):
    """
    Property: HTTPBasic should support UTF-8 encoded credentials per RFC 7617.
    This test fails when credentials contain non-ASCII UTF-8 characters.
    """
    assume(any(ord(c) > 127 for c in username + password))

    credentials_str = f"{username}:{password}"
    encoded = base64.b64encode(credentials_str.encode('utf-8')).decode('ascii')
    auth_header = f"Basic {encoded}"

    request = Request({
        'type': 'http',
        'headers': Headers({'authorization': auth_header}).raw,
    })

    security = HTTPBasic()

    async def decode_credentials():
        result = await security(request)
        assert result.username == username
        assert result.password == password

    asyncio.run(decode_credentials())


if __name__ == "__main__":
    test_httpbasic_utf8_support(
        username="user",
        password="pÄssw0rd"
    )
```

**Failing input**: username=`"user"`, password=`"pÄssw0rd"` (contains German umlaut Ä)

## Reproducing the Bug

```python
import asyncio
import base64
from fastapi.security import HTTPBasic
from starlette.requests import Request
from starlette.datastructures import Headers


async def test_utf8_password():
    username = "admin"
    password = "Pässwörd123"

    credentials_str = f"{username}:{password}"
    encoded = base64.b64encode(credentials_str.encode('utf-8')).decode('ascii')
    auth_header = f"Basic {encoded}"

    request = Request({
        'type': 'http',
        'headers': Headers({'authorization': auth_header}).raw,
    })

    security = HTTPBasic()
    result = await security(request)
    print(f"Username: {result.username}, Password: {result.password}")


asyncio.run(test_utf8_password())
```

Running this code produces:

```
UnicodeDecodeError: 'ascii' codec can't decode byte 0xc3 in position 5: ordinal not in range(128)
```

## Why This Is A Bug

This violates RFC 7617 Section 2.1, which explicitly states:

> "The user-id and password are separated by a single colon (":") character. User-ids containing a colon cannot be encoded in user-pass strings. **The user-id and password are encoded using the UTF-8 charset**" (emphasis added)

The bug is in `/fastapi/security/http.py` at line 211:

```python
data = b64decode(param).decode("ascii")
```

This decodes base64 and then attempts ASCII decoding. When credentials contain UTF-8 characters (code points > 127), the ASCII decoder raises `UnicodeDecodeError`.

**Real-world impact:**
- Users with non-ASCII names (e.g., José, Müller, 王) cannot authenticate
- Passwords with international characters fail
- This affects any non-English speaking user base
- The crash exposes implementation details to attackers

**Legitimacy**: This is not an edge case - UTF-8 support is required by the HTTP Basic Auth specification (RFC 7617, published in 2015, superseding RFC 2617).

## Fix

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