# Bug Report: FastAPI OAuth2 Bearer Returns Empty String Instead of None

**Target**: `fastapi.security.oauth2.OAuth2PasswordBearer` and `fastapi.security.oauth2.OAuth2AuthorizationCodeBearer`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

When the Authorization header contains only the scheme "Bearer" without a token (e.g., `Authorization: Bearer`), OAuth2PasswordBearer and OAuth2AuthorizationCodeBearer return an empty string `""` instead of `None`, creating inconsistent behavior with the case where the Authorization header is missing entirely.

## Property-Based Test

```python
import asyncio
from hypothesis import given, strategies as st, settings
from fastapi.security.oauth2 import OAuth2PasswordBearer
from starlette.datastructures import Headers


class MockRequest:
    def __init__(self, headers_dict):
        self.headers = Headers(headers_dict)


@given(st.just("Bearer"))
@settings(max_examples=10)
def test_oauth2_with_scheme_only_should_return_none(scheme_only):
    oauth2 = OAuth2PasswordBearer(tokenUrl="/token", auto_error=False)
    request = MockRequest({"Authorization": scheme_only})
    result = asyncio.run(oauth2(request))
    assert result is None
```

**Failing input**: `Authorization: Bearer`

## Reproducing the Bug

```python
import asyncio
from fastapi.security.oauth2 import OAuth2PasswordBearer
from starlette.datastructures import Headers


class MockRequest:
    def __init__(self, headers_dict):
        self.headers = Headers(headers_dict)


oauth2 = OAuth2PasswordBearer(tokenUrl="/token", auto_error=False)

request_missing = MockRequest({})
result_missing = asyncio.run(oauth2(request_missing))
print(f"Missing header: {repr(result_missing)}")

request_empty_token = MockRequest({"Authorization": "Bearer"})
result_empty_token = asyncio.run(oauth2(request_empty_token))
print(f"Empty token: {repr(result_empty_token)}")

print(f"Inconsistent: {result_missing} != {result_empty_token}")
```

Output:
```
Missing header: None
Empty token: ''
Inconsistent: None != ''
```

## Why This Is A Bug

1. **Violates documented behavior**: The `auto_error=False` parameter is documented to return `None` when authentication is not available (oauth2.py:448-450). An empty token string is not valid authentication.

2. **Inconsistent behavior**: Missing Authorization header returns `None`, but Authorization header with empty token returns `""`. Both cases represent unavailable authentication.

3. **Invalid token**: An empty string is not a valid OAuth2 bearer token according to RFC 6750.

4. **Breaks user code**: Code that checks `if token:` will incorrectly treat `""` as falsy, but code checking `if token is not None:` will incorrectly treat it as a valid token.

## Fix

```diff
--- a/fastapi/security/oauth2.py
+++ b/fastapi/security/oauth2.py
@@ -488,7 +488,7 @@ class OAuth2PasswordBearer(OAuth2):
     async def __call__(self, request: Request) -> Optional[str]:
         authorization = request.headers.get("Authorization")
         scheme, param = get_authorization_scheme_param(authorization)
-        if not authorization or scheme.lower() != "bearer":
+        if not authorization or scheme.lower() != "bearer" or not param:
             if self.auto_error:
                 raise HTTPException(
                     status_code=HTTP_401_UNAUTHORIZED,
@@ -598,7 +598,7 @@ class OAuth2AuthorizationCodeBearer(OAuth2):
     async def __call__(self, request: Request) -> Optional[str]:
         authorization = request.headers.get("Authorization")
         scheme, param = get_authorization_scheme_param(authorization)
-        if not authorization or scheme.lower() != "bearer":
+        if not authorization or scheme.lower() != "bearer" or not param:
             if self.auto_error:
                 raise HTTPException(
                     status_code=HTTP_401_UNAUTHORIZED,
```