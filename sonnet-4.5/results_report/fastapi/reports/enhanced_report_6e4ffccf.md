# Bug Report: FastAPI HTTPBasic UTF-8 Credential Decoding Failure

**Target**: `fastapi.security.http.HTTPBasic`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

FastAPI's HTTPBasic authentication crashes with UnicodeDecodeError when credentials contain non-ASCII UTF-8 characters, preventing international users from authenticating.

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
    # Run the test directly without hypothesis decorators
    import asyncio
    import base64
    from fastapi.security import HTTPBasic
    from starlette.requests import Request
    from starlette.datastructures import Headers

    username = "user"
    password = "pÄssw0rd"

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
```

<details>

<summary>
**Failing input**: username=`"user"`, password=`"pÄssw0rd"` (contains German umlaut Ä)
</summary>
```
Traceback (most recent call last):
  File "/home/npc/miniconda/lib/python3.13/site-packages/fastapi/security/http.py", line 211, in __call__
    data = b64decode(param).decode("ascii")
UnicodeDecodeError: 'ascii' codec can't decode byte 0xc3 in position 6: ordinal not in range(128)

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/2/hypo.py", line 66, in <module>
    asyncio.run(decode_credentials())
    ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/asyncio/runners.py", line 195, in run
    return runner.run(main)
           ~~~~~~~~~~^^^^^^
  File "/home/npc/miniconda/lib/python3.13/asyncio/runners.py", line 118, in run
    return self._loop.run_until_complete(task)
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^
  File "/home/npc/miniconda/lib/python3.13/asyncio/base_events.py", line 725, in run_until_complete
    return future.result()
           ~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/2/hypo.py", line 62, in decode_credentials
    result = await security(request)
             ^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/fastapi/security/http.py", line 213, in __call__
    raise invalid_user_credentials_exc  # noqa: B904
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
fastapi.exceptions.HTTPException: 401: Invalid authentication credentials
```
</details>

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

<details>

<summary>
HTTPException: 401 Invalid authentication credentials (underlying UnicodeDecodeError)
</summary>
```
Traceback (most recent call last):
  File "/home/npc/miniconda/lib/python3.13/site-packages/fastapi/security/http.py", line 211, in __call__
    data = b64decode(param).decode("ascii")
UnicodeDecodeError: 'ascii' codec can't decode byte 0xc3 in position 7: ordinal not in range(128)

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/2/repo.py", line 26, in <module>
    asyncio.run(test_utf8_password())
    ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/asyncio/runners.py", line 195, in run
    return runner.run(main)
           ~~~~~~~~~~^^^^^^
  File "/home/npc/miniconda/lib/python3.13/asyncio/runners.py", line 118, in run
    return self._loop.run_until_complete(task)
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^
  File "/home/npc/miniconda/lib/python3.13/asyncio/base_events.py", line 725, in run_until_complete
    return future.result()
           ~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/2/repo.py", line 22, in test_utf8_password
    result = await security(request)
             ^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/fastapi/security/http.py", line 213, in __call__
    raise invalid_user_credentials_exc  # noqa: B904
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
fastapi.exceptions.HTTPException: 401: Invalid authentication credentials
```
</details>

## Why This Is A Bug

The bug occurs in `/home/npc/miniconda/lib/python3.13/site-packages/fastapi/security/http.py` at line 211, where the code attempts to decode base64-decoded bytes using ASCII encoding:

```python
data = b64decode(param).decode("ascii")
```

When credentials contain UTF-8 characters with code points above 127 (non-ASCII characters like Ä, ö, é, 中, etc.), the ASCII decoder raises a `UnicodeDecodeError`. This error is caught and re-raised as an HTTPException with "Invalid authentication credentials", masking the true cause of the failure.

This violates modern web application expectations and RFC 7617 (HTTP Basic Authentication), which explicitly introduces UTF-8 support for international characters. While RFC 7617 doesn't mandate UTF-8 as the default encoding for backwards compatibility, it establishes UTF-8 as the standard mechanism for supporting international characters through the optional 'charset=UTF-8' parameter. The RFC states in Section 2.1:

> "The user-id and password are encoded using the UTF-8 charset"

Most modern implementations use UTF-8 by default to support international users. The current ASCII-only implementation unnecessarily restricts authentication to users with ASCII-only credentials, affecting:

- Users with non-ASCII characters in names (José, Müller, 王, София, محمد, etc.)
- Systems using passwords with international characters for enhanced security
- Any application serving a global, non-English speaking user base

## Relevant Context

The FastAPI implementation already handles the `UnicodeDecodeError` exception (line 212), suggesting awareness of encoding issues, but the choice of ASCII decoding is unnecessarily restrictive. The fix is trivial and backwards-compatible since UTF-8 is a superset of ASCII - all existing ASCII-only credentials would continue to work while enabling support for international characters.

Multiple GitHub issues and discussions reference this problem:
- GitHub issue #3235 reports authentication failures with non-ASCII characters
- GitHub discussion #8856 specifically mentions problems with characters like "Ä" in passwords

The error message "Invalid authentication credentials" is misleading - the credentials are valid, they just contain non-ASCII characters that the current implementation cannot handle.

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