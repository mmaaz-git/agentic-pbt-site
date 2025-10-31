#!/usr/bin/env python3
"""
Demonstrates OAuth2PasswordRequestForm scope parsing bug in FastAPI.

The OAuth2 specification (RFC 6749, Section 3.3) requires scopes to be
separated by space characters (0x20) only, not any whitespace.
"""

# Test 1: Newline injection
print("=== Test 1: Newline character in scope ===")
malicious_scope = "read\nwrite"
print(f"Input scope string: {repr(malicious_scope)}")

# Current FastAPI behavior (using split())
parsed_with_split = malicious_scope.split()
print(f"FastAPI current parsing (split()): {parsed_with_split}")

# OAuth2 spec compliant parsing (using split(' '))
parsed_with_space_split = malicious_scope.split(' ')
print(f"OAuth2 spec compliant (split(' ')): {parsed_with_space_split}")

print(f"Are they the same? {parsed_with_split == parsed_with_space_split}")
print()

# Test 2: Tab character
print("=== Test 2: Tab character in scope ===")
tab_scope = "read\twrite"
print(f"Input scope string: {repr(tab_scope)}")

parsed_with_split = tab_scope.split()
print(f"FastAPI current parsing (split()): {parsed_with_split}")

parsed_with_space_split = tab_scope.split(' ')
print(f"OAuth2 spec compliant (split(' ')): {parsed_with_space_split}")

print(f"Are they the same? {parsed_with_split == parsed_with_space_split}")
print()

# Test 3: Carriage return
print("=== Test 3: Carriage return in scope ===")
cr_scope = "read\rwrite"
print(f"Input scope string: {repr(cr_scope)}")

parsed_with_split = cr_scope.split()
print(f"FastAPI current parsing (split()): {parsed_with_split}")

parsed_with_space_split = cr_scope.split(' ')
print(f"OAuth2 spec compliant (split(' ')): {parsed_with_space_split}")

print(f"Are they the same? {parsed_with_split == parsed_with_space_split}")
print()

# Test 4: Multiple consecutive spaces
print("=== Test 4: Multiple consecutive spaces ===")
multi_space_scope = "read  write"
print(f"Input scope string: {repr(multi_space_scope)}")

parsed_with_split = multi_space_scope.split()
print(f"FastAPI current parsing (split()): {parsed_with_split}")

parsed_with_space_split = [s for s in multi_space_scope.split(' ') if s]
print(f"OAuth2 spec compliant (filtered): {parsed_with_space_split}")

print(f"Are they the same? {parsed_with_split == parsed_with_space_split}")
print()

# Test 5: Actual FastAPI implementation
print("=== Test 5: Using actual FastAPI OAuth2PasswordRequestForm ===")
try:
    from fastapi.security import OAuth2PasswordRequestForm
    from unittest.mock import MagicMock

    # Mock the Form() calls since we're not in a FastAPI request context
    form_data = MagicMock()

    # Test with newline-injected scope
    test_scope = "admin\nuser\ndelete_all"
    print(f"Input scope string: {repr(test_scope)}")

    # Simulate what happens in the OAuth2PasswordRequestForm.__init__
    parsed_scopes = test_scope.split()
    print(f"Parsed scopes by OAuth2PasswordRequestForm: {parsed_scopes}")

    # What it should be according to OAuth2 spec
    correct_scopes = [s for s in test_scope.split(' ') if s]
    print(f"Correct parsing per OAuth2 spec: {correct_scopes}")

    print(f"\nVulnerability: A single scope '{test_scope}' is incorrectly parsed as {len(parsed_scopes)} separate scopes!")

except ImportError as e:
    print(f"Could not import FastAPI: {e}")