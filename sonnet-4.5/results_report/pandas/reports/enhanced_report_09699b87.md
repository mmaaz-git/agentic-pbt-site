# Bug Report: fastapi.security.utils get_authorization_scheme_param Multiple Spaces Break Authentication

**Target**: `fastapi.security.utils.get_authorization_scheme_param`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `get_authorization_scheme_param` function incorrectly handles Authorization headers with multiple spaces between the scheme and credentials, causing leading whitespace to be included in the returned parameter value, which breaks authentication for OAuth2 and HTTP Bearer schemes.

## Property-Based Test

```python
#!/usr/bin/env python3
"""Property-based test for get_authorization_scheme_param bug with multiple spaces."""

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


if __name__ == "__main__":
    test_get_authorization_scheme_param_multiple_spaces()
```

<details>

<summary>
**Failing input**: `scheme_input='0', param_input='0', num_spaces=2`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/32/hypo.py", line 22, in <module>
    test_get_authorization_scheme_param_multiple_spaces()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/32/hypo.py", line 9, in test_get_authorization_scheme_param_multiple_spaces
    def test_get_authorization_scheme_param_multiple_spaces(scheme_input, param_input, num_spaces):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/32/hypo.py", line 18, in test_get_authorization_scheme_param_multiple_spaces
    assert param == param_input
           ^^^^^^^^^^^^^^^^^^^^
AssertionError
Falsifying example: test_get_authorization_scheme_param_multiple_spaces(
    # The test always failed when commented parts were varied together.
    scheme_input='0',  # or any other generated value
    param_input='0',  # or any other generated value
    num_spaces=2,  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""Minimal reproduction of the get_authorization_scheme_param bug with multiple spaces."""

from fastapi.security.utils import get_authorization_scheme_param

# Test with single space (works correctly)
authorization_header_single = "Bearer my-token"
scheme_single, param_single = get_authorization_scheme_param(authorization_header_single)

print("=== Single Space Test ===")
print(f"Input: {repr(authorization_header_single)}")
print(f"Scheme: {repr(scheme_single)}")
print(f"Param: {repr(param_single)}")
assert scheme_single == "Bearer"
assert param_single == "my-token"
print("✓ Single space test passed\n")

# Test with double space (fails)
authorization_header_double = "Bearer  my-token"
scheme_double, param_double = get_authorization_scheme_param(authorization_header_double)

print("=== Double Space Test ===")
print(f"Input: {repr(authorization_header_double)}")
print(f"Scheme: {repr(scheme_double)}")
print(f"Param: {repr(param_double)}")

print(f"\nExpected param: 'my-token'")
print(f"Actual param: {repr(param_double)}")

try:
    assert param_double == "my-token", f"Expected 'my-token', got {repr(param_double)}"
    print("✓ Double space test passed")
except AssertionError as e:
    print(f"✗ Assertion Error: {e}")

# Test with triple space (also fails)
authorization_header_triple = "Bearer   my-token"
scheme_triple, param_triple = get_authorization_scheme_param(authorization_header_triple)

print("\n=== Triple Space Test ===")
print(f"Input: {repr(authorization_header_triple)}")
print(f"Scheme: {repr(scheme_triple)}")
print(f"Param: {repr(param_triple)}")

try:
    assert param_triple == "my-token", f"Expected 'my-token', got {repr(param_triple)}"
    print("✓ Triple space test passed")
except AssertionError as e:
    print(f"✗ Assertion Error: {e}")

# Real-world impact demonstration with OAuth2
print("\n=== Real-World Impact: OAuth2PasswordBearer ===")
from fastapi.security import OAuth2PasswordBearer
import asyncio
from starlette.datastructures import Headers

class MockRequest:
    def __init__(self, auth_header):
        self.headers = Headers({"authorization": auth_header})

oauth2 = OAuth2PasswordBearer(tokenUrl="/token", auto_error=False)

token_normal = asyncio.run(oauth2(MockRequest("Bearer token123")))
token_double = asyncio.run(oauth2(MockRequest("Bearer  token123")))
token_triple = asyncio.run(oauth2(MockRequest("Bearer   token123")))

print(f"Normal (1 space): {repr(token_normal)}")
print(f"Double (2 spaces): {repr(token_double)}")
print(f"Triple (3 spaces): {repr(token_triple)}")
print(f"\nTokens match:")
print(f"  Normal == Double: {token_normal == token_double}")
print(f"  Normal == Triple: {token_normal == token_triple}")
print(f"  Double == Triple: {token_double == token_triple}")
```

<details>

<summary>
AssertionError: Multiple spaces cause leading whitespace in parameter
</summary>
```
=== Single Space Test ===
Input: 'Bearer my-token'
Scheme: 'Bearer'
Param: 'my-token'
✓ Single space test passed

=== Double Space Test ===
Input: 'Bearer  my-token'
Scheme: 'Bearer'
Param: ' my-token'

Expected param: 'my-token'
Actual param: ' my-token'
✗ Assertion Error: Expected 'my-token', got ' my-token'

=== Triple Space Test ===
Input: 'Bearer   my-token'
Scheme: 'Bearer'
Param: '  my-token'
✗ Assertion Error: Expected 'my-token', got '  my-token'

=== Real-World Impact: OAuth2PasswordBearer ===
Normal (1 space): 'token123'
Double (2 spaces): ' token123'
Triple (3 spaces): '  token123'

Tokens match:
  Normal == Double: False
  Normal == Triple: False
  Double == Triple: False
```
</details>

## Why This Is A Bug

This violates RFC 7235 (HTTP Authentication) specification, which explicitly allows "1*SP" (one or more spaces) between the auth-scheme and credentials. The current implementation uses `str.partition(" ")` which only splits on the first space character, leaving any additional spaces as part of the parameter value.

According to RFC 7235 section 2.1, the Authorization header format is:
```
Authorization = credentials
credentials = auth-scheme [ 1*SP ( token68 / [ ( "," / auth-param ) *( OWS "," [ OWS auth-param ] ) ] ) ]
```

The "1*SP" notation means one or more spaces are allowed and should be handled correctly. All of these should be equivalent:
- `"Bearer token123"` (1 space) → param should be `"token123"`
- `"Bearer  token123"` (2 spaces) → param should be `"token123"`
- `"Bearer   token123"` (3 spaces) → param should be `"token123"`

However, the current implementation returns:
- `"Bearer token123"` → param = `"token123"` ✓
- `"Bearer  token123"` → param = `" token123"` ✗ (includes leading space)
- `"Bearer   token123"` → param = `"  token123"` ✗ (includes two leading spaces)

This causes authentication failures because the token `" token123"` (with leading space) is different from `"token123"`. Real-world scenarios where this occurs:
1. HTTP clients that accidentally include extra spaces
2. Proxy servers or middleware that normalize whitespace
3. Manual testing with tools that auto-format headers
4. Copy-paste errors during development

## Relevant Context

The bug is in `/fastapi/security/utils.py` in the `get_authorization_scheme_param` function:
- Source: https://github.com/tiangolo/fastapi/blob/master/fastapi/security/utils.py
- The function is used internally by FastAPI's OAuth2PasswordBearer and HTTPBearer security schemes
- RFC 7235 specification: https://datatracker.ietf.org/doc/html/rfc7235#section-2.1

The function's current implementation:
```python
def get_authorization_scheme_param(
    authorization_header_value: Optional[str],
) -> Tuple[str, str]:
    if not authorization_header_value:
        return "", ""
    scheme, _, param = authorization_header_value.partition(" ")
    return scheme, param
```

The `str.partition(" ")` method splits on only the first occurrence of a space, so when there are multiple spaces, the remaining spaces become part of the parameter value.

## Proposed Fix

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