# Bug Report: fastapi.security.HTTPBasic realm Parameter Not Properly Escaped in WWW-Authenticate Header

**Target**: `fastapi.security.http.HTTPBasic`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `realm` parameter in FastAPI's `HTTPBasic` authentication is directly interpolated into the `WWW-Authenticate` header without proper escaping, violating RFC 7235 and RFC 7230 specifications and producing malformed HTTP headers when the realm contains special characters like quotes, backslashes, or newlines.

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

<details>

<summary>
**Failing input**: `realm='\n'`
</summary>
```
============================= test session starts ==============================
platform linux -- Python 3.13.2, pytest-8.4.1, pluggy-1.5.0 -- /home/npc/miniconda/bin/python3
cachedir: .pytest_cache
hypothesis profile 'default'
rootdir: /home/npc/pbt/agentic-pbt/worker_/16
plugins: anyio-4.9.0, hypothesis-6.139.1, asyncio-1.2.0, langsmith-0.4.29
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 1 item

hypo.py::test_http_basic_realm_must_be_properly_escaped FAILED           [100%]

=================================== FAILURES ===================================
________________ test_http_basic_realm_must_be_properly_escaped ________________

realm = '\n'

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
>           await basic(request)

hypo.py:24:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

self = <fastapi.security.http.HTTPBasic object at 0x746423d59bd0>
request = <starlette.requests.Request object at 0x746423d59a90>

    async def __call__(  # type: ignore
        self, request: Request
    ) -> Optional[HTTPBasicCredentials]:
        authorization = request.headers.get("Authorization")
        scheme, param = get_authorization_scheme_param(authorization)
        if self.realm:
            unauthorized_headers = {"WWW-Authenticate": f'Basic realm="{self.realm}"'}
        else:
            unauthorized_headers = {"WWW-Authenticate": "Basic"}
        if not authorization or scheme.lower() != "basic":
            if self.auto_error:
>               raise HTTPException(
                    status_code=HTTP_401_UNAUTHORIZED,
                    detail="Not authenticated",
                    headers=unauthorized_headers,
                )
E               fastapi.exceptions.HTTPException: 401: Not authenticated

../../envs/fastapi_env/lib/python3.13/site-packages/fastapi/security/http.py:198: HTTPException

During handling of the above exception, another exception occurred:

    @pytest.mark.asyncio
>   @given(st.text(alphabet=st.characters(min_codepoint=1, max_codepoint=127)))
                   ^^^

hypo.py:12:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
/home/npc/miniconda/lib/python3.13/site-packages/pytest_asyncio/plugin.py:721: in inner
    runner.run(coro, context=context)
/home/npc/miniconda/lib/python3.13/asyncio/runners.py:118: in run
    return self._loop.run_until_complete(task)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
/home/npc/miniconda/lib/python3.13/asyncio/base_events.py:725: in run_until_complete
    return future.result()
           ^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

realm = '\n'

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
>               assert '\n' not in www_authenticate and '\r' not in www_authenticate, \
                    f"Newlines must not appear in HTTP headers but got: {repr(www_authenticate)}"
E               AssertionError: Newlines must not appear in HTTP headers but got: 'Basic realm="\n"'
E               assert ('\n' not in 'Basic realm="\n"'
E
E                 '\n' is contained here:
E                 ?              ^
E                   Basic realm="
E                 ?              ^
E                   ")
E               Falsifying example: test_http_basic_realm_must_be_properly_escaped(
E                   realm='\n',
E               )

hypo.py:37: AssertionError
=========================== short test summary info ============================
FAILED hypo.py::test_http_basic_realm_must_be_properly_escaped - AssertionErr...
============================== 1 failed in 0.33s ===============================
```
</details>

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/fastapi_env/lib/python3.13/site-packages')

import asyncio
from fastapi.security.http import HTTPBasic
from fastapi.exceptions import HTTPException
from starlette.requests import Request


async def reproduce():
    # Test case 1: Realm with quotes
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
        print("Test 1: Realm with quotes")
        print(f"Input realm:    'My\"Realm'")
        print(f"Actual header:  {www_auth}")
        print(f"Expected:       Basic realm=\"My\\\"Realm\"")
        print()

    # Test case 2: Realm with backslashes
    basic2 = HTTPBasic(realm='My\\Realm', auto_error=True)

    try:
        await basic2(request)
    except HTTPException as e:
        www_auth = e.headers.get("WWW-Authenticate")
        print("Test 2: Realm with backslashes")
        print(f"Input realm:    'My\\\\Realm'")
        print(f"Actual header:  {www_auth}")
        print(f"Expected:       Basic realm=\"My\\\\Realm\"")
        print()

    # Test case 3: Realm with both quotes and backslashes
    basic3 = HTTPBasic(realm='Admin\\"s Area', auto_error=True)

    try:
        await basic3(request)
    except HTTPException as e:
        www_auth = e.headers.get("WWW-Authenticate")
        print("Test 3: Realm with both quotes and backslashes")
        print(f"Input realm:    'Admin\\\\\"s Area'")
        print(f"Actual header:  {www_auth}")
        print(f"Expected:       Basic realm=\"Admin\\\\\\\"s Area\"")


asyncio.run(reproduce())
```

<details>

<summary>
Output showing malformed WWW-Authenticate headers
</summary>
```
Test 1: Realm with quotes
Input realm:    'My"Realm'
Actual header:  Basic realm="My"Realm"
Expected:       Basic realm="My\"Realm"

Test 2: Realm with backslashes
Input realm:    'My\\Realm'
Actual header:  Basic realm="My\Realm"
Expected:       Basic realm="My\\Realm"

Test 3: Realm with both quotes and backslashes
Input realm:    'Admin\\"s Area'
Actual header:  Basic realm="Admin\"s Area"
Expected:       Basic realm="Admin\\\"s Area"
```
</details>

## Why This Is A Bug

This implementation violates HTTP protocol specifications in multiple ways:

1. **RFC 7235 Section 2.2** specifies that the realm parameter must be a quoted-string as defined in RFC 7230.

2. **RFC 7230 Section 3.2.6** defines the quoted-string format and explicitly requires:
   - Double quotes (`"`) inside quoted-strings must be escaped with a backslash: `\"`
   - Backslashes (`\`) must themselves be escaped: `\\`
   - Control characters (including newlines `\n` and carriage returns `\r`) are forbidden in HTTP headers

3. The current implementation at line 193 of `/home/npc/pbt/agentic-pbt/envs/fastapi_env/lib/python3.13/site-packages/fastapi/security/http.py` directly interpolates the realm value without any escaping:
   ```python
   unauthorized_headers = {"WWW-Authenticate": f'Basic realm="{self.realm}"'}
   ```

This causes:
- **Malformed HTTP headers** when realm contains quotes, producing syntactically invalid header values like `Basic realm="My"Realm"`
- **Incorrect escaping** when realm contains backslashes, losing the backslash character
- **Potential header injection vulnerabilities** when realm contains newlines, though the impact may be limited by downstream sanitization

## Relevant Context

The bug affects any FastAPI application using HTTP Basic authentication with custom realm values. While many HTTP clients are lenient and may still parse malformed headers, standards-compliant clients may fail to authenticate properly.

Common real-world scenarios where this bug manifests:
- Realm names with apostrophes: `Admin's Portal`
- Realm names with quotes: `"Production" Environment`
- File path-like realms: `C:\Users\Admin`
- Multi-line realm descriptions (though these shouldn't be used)

The FastAPI documentation for HTTPBasic is at: https://fastapi.tiangolo.com/advanced/security/http-basic-auth/

The relevant HTTP specifications:
- RFC 7235 (HTTP Authentication): https://datatracker.ietf.org/doc/html/rfc7235#section-2.2
- RFC 7230 (HTTP Message Syntax): https://datatracker.ietf.org/doc/html/rfc7230#section-3.2.6

## Proposed Fix

```diff
--- a/fastapi/security/http.py
+++ b/fastapi/security/http.py
@@ -189,8 +189,14 @@ class HTTPBasic(HTTPBase):
     ) -> Optional[HTTPBasicCredentials]:
         authorization = request.headers.get("Authorization")
         scheme, param = get_authorization_scheme_param(authorization)
         if self.realm:
-            unauthorized_headers = {"WWW-Authenticate": f'Basic realm="{self.realm}"'}
+            # Escape realm value for quoted-string per RFC 7230 Section 3.2.6
+            escaped_realm = self.realm.replace("\\", "\\\\").replace('"', '\\"')
+            # Remove any control characters that would make the header invalid
+            escaped_realm = "".join(
+                c for c in escaped_realm
+                if ord(c) >= 32 and ord(c) != 127
+            )
+            unauthorized_headers = {"WWW-Authenticate": f'Basic realm="{escaped_realm}"'}
         else:
             unauthorized_headers = {"WWW-Authenticate": "Basic"}
         if not authorization or scheme.lower() != "basic":
```