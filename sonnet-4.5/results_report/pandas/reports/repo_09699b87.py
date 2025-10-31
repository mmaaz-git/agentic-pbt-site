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