# Bug Report: fastapi.security.utils.get_authorization_scheme_param Multiple Spaces Not Handled Correctly

**Target**: `fastapi.security.utils.get_authorization_scheme_param`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `get_authorization_scheme_param` function incorrectly handles Authorization headers with multiple spaces between the scheme and credentials, causing leading whitespace to be included in the returned parameter value, which breaks authentication for OAuth2 and HTTP authentication schemes.

## Property-Based Test

```python
#!/usr/bin/env python3
"""
Hypothesis property-based test that discovers the bug in
fastapi.security.utils.get_authorization_scheme_param
"""

from hypothesis import given, strategies as st, assume
from fastapi.security.utils import get_authorization_scheme_param


@given(st.text(min_size=1), st.text(min_size=1), st.integers(min_value=2, max_value=10))
def test_get_authorization_scheme_param_multiple_spaces(scheme_input, param_input, num_spaces):
    assume(" " not in scheme_input)
    assume(" " not in param_input)

    authorization = f"{scheme_input}{' ' * num_spaces}{param_input}"

    scheme, param = get_authorization_scheme_param(authorization)

    assert scheme == scheme_input
    assert param == param_input, f"Expected {repr(param_input)}, got {repr(param)}"


if __name__ == "__main__":
    # Run the test
    test_get_authorization_scheme_param_multiple_spaces()
```

<details>

<summary>
**Failing input**: `scheme_input='0', param_input='0', num_spaces=2`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/12/hypo.py", line 26, in <module>
    test_get_authorization_scheme_param_multiple_spaces()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/12/hypo.py", line 12, in test_get_authorization_scheme_param_multiple_spaces
    def test_get_authorization_scheme_param_multiple_spaces(scheme_input, param_input, num_spaces):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/12/hypo.py", line 21, in test_get_authorization_scheme_param_multiple_spaces
    assert param == param_input, f"Expected {repr(param_input)}, got {repr(param)}"
           ^^^^^^^^^^^^^^^^^^^^
AssertionError: Expected '0', got ' 0'
Falsifying example: test_get_authorization_scheme_param_multiple_spaces(
    scheme_input='0',
    param_input='0',
    num_spaces=2,
)
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""
Minimal reproduction of the FastAPI get_authorization_scheme_param bug
with multiple spaces in Authorization header.
"""

from fastapi.security.utils import get_authorization_scheme_param
from fastapi.security import OAuth2PasswordBearer
import asyncio
from starlette.datastructures import Headers

print("="*60)
print("Test 1: Direct function call with double space")
print("="*60)

# Test with double space between Bearer and token
authorization_header = "Bearer  my-token"
scheme, param = get_authorization_scheme_param(authorization_header)

print(f"Input: {repr(authorization_header)}")
print(f"Scheme: {repr(scheme)}")
print(f"Param: {repr(param)}")
print(f"Expected param: 'my-token'")
print(f"Actual param:   {repr(param)}")
print(f"Leading space present: {param.startswith(' ')}")
print()

# Verify the assertion fails
try:
    assert param == "my-token", f"Expected 'my-token', got {repr(param)}"
    print("✓ Assertion passed")
except AssertionError as e:
    print(f"✗ Assertion failed: {e}")

print("\n" + "="*60)
print("Test 2: OAuth2PasswordBearer impact")
print("="*60)

# Mock request class to simulate FastAPI request
class MockRequest:
    def __init__(self, auth_header):
        self.headers = Headers({"authorization": auth_header})

oauth2 = OAuth2PasswordBearer(tokenUrl="/token", auto_error=False)

# Test with single space (normal case)
token1 = asyncio.run(oauth2(MockRequest("Bearer token123")))

# Test with double space (bug case)
token2 = asyncio.run(oauth2(MockRequest("Bearer  token123")))

# Test with triple space
token3 = asyncio.run(oauth2(MockRequest("Bearer   token123")))

print(f"Single space: {repr(token1)}")
print(f"Double space: {repr(token2)}")
print(f"Triple space: {repr(token3)}")
print()
print(f"token1 == 'token123': {token1 == 'token123'}")
print(f"token2 == 'token123': {token2 == 'token123'}")
print(f"token3 == 'token123': {token3 == 'token123'}")
print()
print(f"Tokens match (1 vs 2): {token1 == token2}")
print(f"Tokens match (1 vs 3): {token1 == token3}")

print("\n" + "="*60)
print("Test 3: Multiple authorization schemes")
print("="*60)

test_cases = [
    ("Bearer token", "Normal single space"),
    ("Bearer  token", "Double space"),
    ("Bearer   token", "Triple space"),
    ("Bearer\ttoken", "Tab character"),
    ("Basic  dXNlcjpwYXNz", "Basic auth with double space"),
    ("Digest  username=foo", "Digest auth with double space"),
]

for auth_header, description in test_cases:
    scheme, param = get_authorization_scheme_param(auth_header)
    print(f"{description:30} | scheme={repr(scheme):10} | param={repr(param)}")

print("\n" + "="*60)
print("Summary")
print("="*60)
print("The bug: get_authorization_scheme_param does not strip leading")
print("whitespace from the parameter after splitting on the first space.")
print("This violates RFC 7235 which allows '1*SP' (one or more spaces)")
print("between the auth-scheme and credentials.")
```

<details>

<summary>
Output shows tokens with leading spaces are incorrectly returned
</summary>
```
============================================================
Test 1: Direct function call with double space
============================================================
Input: 'Bearer  my-token'
Scheme: 'Bearer'
Param: ' my-token'
Expected param: 'my-token'
Actual param:   ' my-token'
Leading space present: True

✗ Assertion failed: Expected 'my-token', got ' my-token'

============================================================
Test 2: OAuth2PasswordBearer impact
============================================================
Single space: 'token123'
Double space: ' token123'
Triple space: '  token123'

token1 == 'token123': True
token2 == 'token123': False
token3 == 'token123': False

Tokens match (1 vs 2): False
Tokens match (1 vs 3): False

============================================================
Test 3: Multiple authorization schemes
============================================================
Normal single space            | scheme='Bearer'   | param='token'
Double space                   | scheme='Bearer'   | param=' token'
Triple space                   | scheme='Bearer'   | param='  token'
Tab character                  | scheme='Bearer\ttoken' | param=''
Basic auth with double space   | scheme='Basic'    | param=' dXNlcjpwYXNz'
Digest auth with double space  | scheme='Digest'   | param=' username=foo'

============================================================
Summary
============================================================
The bug: get_authorization_scheme_param does not strip leading
whitespace from the parameter after splitting on the first space.
This violates RFC 7235 which allows '1*SP' (one or more spaces)
between the auth-scheme and credentials.
```
</details>

## Why This Is A Bug

This is a violation of RFC 7235 (HTTP/1.1 Authentication) which explicitly defines the Authorization header format as:

```
credentials = auth-scheme [ 1*SP ( token68 / #auth-param ) ]
```

The notation `1*SP` means "one or more space characters" in ABNF notation, indicating that multiple spaces between the auth-scheme and credentials are valid and should be normalized.

The current implementation in `/home/npc/pbt/agentic-pbt/envs/fastapi_env/lib/python3.13/site-packages/fastapi/security/utils.py` uses `str.partition(" ")` which only splits on the first space, leaving any additional spaces as part of the credential value. This causes:

1. **Authentication failures**: Tokens like `"Bearer  token123"` return `" token123"` instead of `"token123"`, causing token validation to fail
2. **RFC non-compliance**: The implementation doesn't conform to the HTTP authentication standard
3. **Real-world impact**: HTTP clients, proxies, or middleware that normalize whitespace or accidentally include extra spaces will fail authentication

The bug affects all FastAPI security classes that use this function:
- `OAuth2PasswordBearer`
- `OAuth2AuthorizationCodeBearer`
- `HTTPBearer`
- `HTTPDigest`

Note: `HTTPBasic` is not affected because base64 decoding automatically strips whitespace.

## Relevant Context

The function is located at line 4-10 in `fastapi/security/utils.py`:

```python
def get_authorization_scheme_param(
    authorization_header_value: Optional[str],
) -> Tuple[str, str]:
    if not authorization_header_value:
        return "", ""
    scheme, _, param = authorization_header_value.partition(" ")
    return scheme, param
```

The OAuth2PasswordBearer class uses this function in its `__call__` method (line 488-500 in `fastapi/security/oauth2.py`):

```python
async def __call__(self, request: Request) -> Optional[str]:
    authorization = request.headers.get("Authorization")
    scheme, param = get_authorization_scheme_param(authorization)
    if not authorization or scheme.lower() != "bearer":
        if self.auto_error:
            raise HTTPException(...)
        else:
            return None
    return param  # Returns the token with leading spaces if present
```

RFC 7235 reference: https://datatracker.ietf.org/doc/html/rfc7235#section-2.1

## Proposed Fix

```diff
--- a/fastapi/security/utils.py
+++ b/fastapi/security/utils.py
@@ -7,4 +7,4 @@ def get_authorization_scheme_param(
     if not authorization_header_value:
         return "", ""
     scheme, _, param = authorization_header_value.partition(" ")
-    return scheme, param
+    return scheme, param.lstrip()
```