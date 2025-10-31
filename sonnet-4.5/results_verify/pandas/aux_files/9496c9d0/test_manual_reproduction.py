#!/usr/bin/env python3
"""Manual reproduction from the bug report"""

from fastapi.security.utils import get_authorization_scheme_param

# Test 1: Basic test with double space
print("=== Test 1: Basic reproduction ===")
authorization_header = "Bearer  my-token"
scheme, param = get_authorization_scheme_param(authorization_header)

print(f"Input: {repr(authorization_header)}")
print(f"Scheme: {repr(scheme)}")
print(f"Param: {repr(param)}")

try:
    assert scheme == "Bearer"
    print("✓ Scheme assertion passed")
except AssertionError:
    print("✗ Scheme assertion failed")

try:
    assert param == "my-token", f"Expected 'my-token', got {repr(param)}"
    print("✓ Param assertion passed")
except AssertionError as e:
    print(f"✗ Param assertion failed: {e}")

# Test 2: Multiple spaces test
print("\n=== Test 2: Various numbers of spaces ===")
for num_spaces in range(1, 5):
    authorization = f"Bearer{' ' * num_spaces}token123"
    scheme, param = get_authorization_scheme_param(authorization)
    print(f"{num_spaces} space(s): Authorization: {repr(authorization)}")
    print(f"  → Scheme: {repr(scheme)}, Param: {repr(param)}")
    if num_spaces > 1:
        expected_leading_spaces = ' ' * (num_spaces - 1)
        print(f"  → Param has {len(param) - len('token123')} leading space(s)")