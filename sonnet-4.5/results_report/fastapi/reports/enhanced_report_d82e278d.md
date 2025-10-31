# Bug Report: FastAPI OAuth2PasswordBearer Returns Empty String for Bearer-Only Headers Instead of None

**Target**: `fastapi.security.oauth2.OAuth2PasswordBearer` and `fastapi.security.oauth2.OAuth2AuthorizationCodeBearer`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

OAuth2PasswordBearer and OAuth2AuthorizationCodeBearer return an empty string `""` when the Authorization header contains only "Bearer" without a token, creating an inconsistency with the documented behavior of returning `None` when authentication is not available (auto_error=False).

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
    assert result is None, f"Expected None for Authorization: '{scheme_only}', but got {repr(result)}"


if __name__ == "__main__":
    # Run the property-based test
    test_oauth2_with_scheme_only_should_return_none()
```

<details>

<summary>
**Failing input**: `Authorization: Bearer`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/37/hypo.py", line 23, in <module>
    test_oauth2_with_scheme_only_should_return_none()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/37/hypo.py", line 13, in test_oauth2_with_scheme_only_should_return_none
    @settings(max_examples=10)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/37/hypo.py", line 18, in test_oauth2_with_scheme_only_should_return_none
    assert result is None, f"Expected None for Authorization: '{scheme_only}', but got {repr(result)}"
           ^^^^^^^^^^^^^^
AssertionError: Expected None for Authorization: 'Bearer', but got ''
Falsifying example: test_oauth2_with_scheme_only_should_return_none(
    scheme_only='Bearer',
)
```
</details>

## Reproducing the Bug

```python
import asyncio
from fastapi.security.oauth2 import OAuth2PasswordBearer
from starlette.datastructures import Headers


class MockRequest:
    def __init__(self, headers_dict):
        self.headers = Headers(headers_dict)


oauth2 = OAuth2PasswordBearer(tokenUrl="/token", auto_error=False)

# Test 1: Missing Authorization header
request_missing = MockRequest({})
result_missing = asyncio.run(oauth2(request_missing))
print(f"Missing header: {repr(result_missing)}")

# Test 2: Authorization header with "Bearer" but no token
request_empty_token = MockRequest({"Authorization": "Bearer"})
result_empty_token = asyncio.run(oauth2(request_empty_token))
print(f"Empty token (Bearer only): {repr(result_empty_token)}")

# Test 3: Authorization header with "Bearer " (with trailing space)
request_empty_token_space = MockRequest({"Authorization": "Bearer "})
result_empty_token_space = asyncio.run(oauth2(request_empty_token_space))
print(f"Empty token (Bearer with space): {repr(result_empty_token_space)}")

# Test 4: Authorization header with valid token
request_valid_token = MockRequest({"Authorization": "Bearer token123"})
result_valid_token = asyncio.run(oauth2(request_valid_token))
print(f"Valid token: {repr(result_valid_token)}")

# Show the inconsistency
print(f"\nInconsistency detected:")
print(f"Missing header returns: {repr(result_missing)}")
print(f"'Bearer' without token returns: {repr(result_empty_token)}")
print(f"These should be the same (both None) but are different: {result_missing != result_empty_token}")
```

<details>

<summary>
Output demonstrating inconsistent behavior
</summary>
```
Missing header: None
Empty token (Bearer only): ''
Empty token (Bearer with space): ''
Valid token: 'token123'

Inconsistency detected:
Missing header returns: None
'Bearer' without token returns: ''
These should be the same (both None) but are different: True
```
</details>

## Why This Is A Bug

This behavior violates the documented contract and creates inconsistencies that can lead to subtle bugs in user code:

1. **Violates documented behavior**: The `auto_error=False` parameter is documented in the OAuth2PasswordBearer class (lines 440-458 in oauth2.py) to return `None` when "the HTTP Authorization header is not available" or when authentication fails. An Authorization header with "Bearer" but no token clearly represents unavailable authentication, as there is no valid token present.

2. **Type hint mismatch**: The return type is annotated as `Optional[str]` (line 488 in oauth2.py), suggesting that `None` is the intended sentinel value for "no authentication available," not an empty string. This creates confusion for developers using type checking.

3. **Inconsistent behavior across equivalent cases**: Three semantically equivalent cases behave differently:
   - No Authorization header → returns `None`
   - Authorization header with wrong scheme (e.g., "Basic") → returns `None`
   - Authorization header with "Bearer" but no token → returns `""`

   All three cases represent "authentication is not available," yet only the third returns an empty string.

4. **Breaks user code patterns**: Different code patterns handle this inconsistently:
   - `if token:` treats both `None` and `""` as falsy (works by accident)
   - `if token is not None:` incorrectly treats `""` as a valid token (BUG!)
   - `if token is None:` fails to catch the empty token case

5. **Security implications**: In security-critical authentication code, predictable and consistent behavior is essential. The inconsistency could lead to authentication bypass vulnerabilities if developers assume `None` is the only indicator of missing authentication.

## Relevant Context

The root cause lies in the interaction between two functions:

1. **`get_authorization_scheme_param`** (in `fastapi/security/utils.py`):
   - When given "Bearer", it returns `("Bearer", "")`
   - The empty string for the param is the source of the issue

2. **`OAuth2PasswordBearer.__call__`** (line 488-500 in `oauth2.py`):
   - Checks: `if not authorization or scheme.lower() != "bearer"`
   - When Authorization is "Bearer":
     - `authorization = "Bearer"` (truthy)
     - `scheme = "Bearer"`, `param = ""`
     - Condition evaluates to False
     - Returns `param` which is `""`

The same issue affects `OAuth2AuthorizationCodeBearer.__call__` (lines 598-610).

RFC 6750 (OAuth 2.0 Bearer Token Usage) defines the syntax as `Authorization: Bearer <token>` where the token must contain at least one character. An empty token violates this specification.

## Proposed Fix

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