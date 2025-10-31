# Bug Report: fastapi.security.HTTPBasic realm Parameter Not Escaped

**Target**: `fastapi.security.http.HTTPBasic`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `realm` parameter in `HTTPBasic` is inserted directly into the `WWW-Authenticate` HTTP header without proper escaping. According to RFC 7235, realm values must be quoted-strings where quotes (`"`) and backslashes (`\`) must be escaped with backslashes. The current implementation violates this specification, producing malformed headers.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/fastapi_env/lib/python3.13/site-packages')

import pytest
from hypothesis import given, strategies as st
from fastapi.security.http import HTTPBasic
from fastapi.exceptions import HTTPException
from starlette.requests import Request


@pytest.mark.asyncio
@given(st.text(alphabet=st.characters(min_codepoint=1, max_codepoint=127)))
async def test_http_basic_realm_must_be_properly_escaped(realm):
    scope = {
        "type": "http",
        "method": "GET",
        "headers": [],
    }
    request = Request(scope)

    basic = HTTPBasic(realm=realm, auto_error=True)

    try:
        await basic(request)
    except HTTPException as exc:
        www_authenticate = exc.headers.get("WWW-Authenticate")

        if '"' in realm:
            assert '\\"' in www_authenticate, \
                f"Quotes in realm must be escaped but got: {www_authenticate}"

        if '\\' in realm:
            assert '\\\\' in www_authenticate, \
                f"Backslashes in realm must be escaped but got: {www_authenticate}"

        if '\n' in realm or '\r' in realm:
            assert '\n' not in www_authenticate and '\r' not in www_authenticate, \
                f"Newlines must not appear in HTTP headers but got: {repr(www_authenticate)}"
```

**Failing input**: `realm='test"value'` produces `Basic realm="test"value"` instead of `Basic realm="test\"value"`

## Reproducing the Bug

```python
import asyncio
from fastapi.security.http import HTTPBasic
from fastapi.exceptions import HTTPException
from starlette.requests import Request


async def reproduce():
    scope = {
        "type": "http",
        "method": "GET",
        "headers": [],
    }
    request = Request(scope)

    basic = HTTPBasic(realm='My"Realm', auto_error=True)

    try:
        await basic(request)
    except HTTPException as e:
        www_auth = e.headers.get("WWW-Authenticate")
        print(f"Actual:   {www_auth}")
        print(f"Expected: Basic realm=\"My\\\"Realm\"")


asyncio.run(reproduce())
```

## Why This Is A Bug

According to [RFC 7235 Section 2.2](https://datatracker.ietf.org/doc/html/rfc7235#section-2.2), the realm parameter is a quoted-string. RFC 7230 Section 3.2.6 defines quoted-string format, which requires:
- Quotes (`"`) inside quoted-strings must be escaped as `\"`
- Backslashes (`\`) must be escaped as `\\`
- Control characters including newlines are not allowed

The current implementation at `fastapi/security/http.py:193` directly interpolates the realm value without escaping:

```python
unauthorized_headers = {"WWW-Authenticate": f'Basic realm="{self.realm}"'}
```

This produces:
1. **Malformed headers** when realm contains quotes or backslashes
2. **Potential header injection** if realm contains newlines (though Starlette may sanitize)

## Fix

```diff
--- a/fastapi/security/http.py
+++ b/fastapi/security/http.py
@@ -1,5 +1,6 @@
 import binascii
 from base64 import b64decode
+import re
 from typing import Optional

 from fastapi.exceptions import HTTPException
@@ -189,8 +190,11 @@ class HTTPBasic(HTTPBase):
     ) -> Optional[HTTPBasicCredentials]:
         authorization = request.headers.get("Authorization")
         scheme, param = get_authorization_scheme_param(authorization)
+        # Escape realm value for quoted-string per RFC 7230 Section 3.2.6
+        escaped_realm = re.sub(r'([\\"])', r'\\\1', self.realm) if self.realm else None
+        escaped_realm = re.sub(r'[\r\n]', '', escaped_realm) if escaped_realm else None
         if self.realm:
-            unauthorized_headers = {"WWW-Authenticate": f'Basic realm="{self.realm}"'}
+            unauthorized_headers = {"WWW-Authenticate": f'Basic realm="{escaped_realm}"'}
         else:
             unauthorized_headers = {"WWW-Authenticate": "Basic"}
         if not authorization or scheme.lower() != "basic":
```