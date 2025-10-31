# Bug Report: fastapi.security get_authorization_scheme_param Multiple Spaces

**Target**: `fastapi.security.utils.get_authorization_scheme_param`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `get_authorization_scheme_param` function incorrectly handles Authorization headers with multiple spaces between the scheme and credentials, causing leading whitespace to be included in the returned parameter value. This breaks authentication for OAuth2PasswordBearer, OAuth2AuthorizationCodeBearer, HTTPBearer, and HTTPDigest when extra spaces are present.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
from fastapi.security.utils import get_authorization_scheme_param


@given(st.text(min_size=1), st.text(min_size=1), st.integers(min_value=2, max_value=10))
def test_get_authorization_scheme_param_multiple_spaces(scheme_input, param_input, num_spaces):
    assume(" " not in scheme_input)
    assume(" " not in param_input)

    authorization = f"{scheme_input}{' ' * num_spaces}{param_input}"

    scheme, param = get_authorization_scheme_param(authorization)

    assert scheme == scheme_input
    assert param == param_input
```

**Failing input**: `scheme_input='0'`, `param_input='0'`, `num_spaces=2`

## Reproducing the Bug

```python
from fastapi.security.utils import get_authorization_scheme_param

authorization_header = "Bearer  my-token"
scheme, param = get_authorization_scheme_param(authorization_header)

print(f"Scheme: {repr(scheme)}")
print(f"Param: {repr(param)}")

assert scheme == "Bearer"
assert param == "my-token", f"Expected 'my-token', got {repr(param)}"
```

Output:
```
Scheme: 'Bearer'
Param: ' my-token'
AssertionError: Expected 'my-token', got ' my-token'
```

Real-world impact:
```python
from fastapi.security import OAuth2PasswordBearer
import asyncio
from starlette.datastructures import Headers

class MockRequest:
    def __init__(self, auth_header):
        self.headers = Headers({"authorization": auth_header})

oauth2 = OAuth2PasswordBearer(tokenUrl="/token", auto_error=False)

token1 = asyncio.run(oauth2(MockRequest("Bearer token123")))
token2 = asyncio.run(oauth2(MockRequest("Bearer  token123")))

print(f"Single space: {repr(token1)}")
print(f"Double space: {repr(token2)}")
print(f"Tokens match: {token1 == token2}")
```

Output:
```
Single space: 'token123'
Double space: ' token123'
Tokens match: False
```

## Why This Is A Bug

According to RFC 7235, the Authorization header format allows one or more spaces (`1*SP`) between the scheme and credentials. The current implementation uses `str.partition(" ")` which splits on only the first space, leaving additional spaces as part of the credentials/token.

This causes authentication to fail when:
1. HTTP clients accidentally include extra spaces in the Authorization header
2. Proxy servers or middleware normalize whitespace
3. Manual testing with tools that auto-format headers

The token " token123" (with leading space) is different from "token123", causing token validation to fail even though the logical value is the same.

Affected classes:
- `OAuth2PasswordBearer`
- `OAuth2AuthorizationCodeBearer`
- `HTTPBearer`
- `HTTPDigest`

Note: `HTTPBasic` is not affected because base64 decoding automatically strips whitespace.

## Fix

```diff
--- a/fastapi/security/utils.py
+++ b/fastapi/security/utils.py
@@ -3,7 +3,7 @@ from typing import Optional, Tuple
 def get_authorization_scheme_param(
     authorization_header_value: Optional[str],
 ) -> Tuple[str, str]:
     if not authorization_header_value:
         return "", ""
     scheme, _, param = authorization_header_value.partition(" ")
-    return scheme, param
+    return scheme, param.lstrip()
```

This change strips leading whitespace from the parameter, ensuring that "Bearer  token" and "Bearer token" both return "token" as the parameter value.