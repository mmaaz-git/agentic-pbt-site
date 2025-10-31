#!/usr/bin/env python3
"""Detailed test to understand scope parsing behavior"""

from fastapi.security import OAuth2PasswordRequestForm, SecurityScopes

def test_split_behavior():
    """Test how Python's split() behaves with and without arguments"""

    test_cases = [
        "read write",           # Normal case
        "read  write",          # Double space
        "read\twrite",          # Tab
        "read\nwrite",          # Newline
        "read\rwrite",          # Carriage return
        "read \t\n\r write",    # Mixed whitespace
        "0\r",                  # Trailing carriage return
        "\r",                   # Just carriage return
        "  read  write  ",      # Leading/trailing spaces
    ]

    print("Testing split() behavior:")
    print("=" * 60)

    for test in test_cases:
        print(f"\nInput: {test!r}")
        print(f"  split():      {test.split()}")
        print(f"  split(' '):   {test.split(' ')}")

        # Test OAuth2PasswordRequestForm
        form = OAuth2PasswordRequestForm(
            username="test",
            password="test",
            scope=test
        )
        print(f"  OAuth2 form.scopes: {form.scopes}")
        print(f"  Rejoined with ' '.join(): {' '.join(form.scopes)!r}")
        print(f"  Round-trip preserved? {test == ' '.join(form.scopes)}")

def test_security_scopes():
    """Test SecurityScopes behavior"""
    print("\n\nTesting SecurityScopes behavior:")
    print("=" * 60)

    test_cases = [
        ["read", "write"],
        ["read", "", "write"],  # Empty string in middle
        ["\r"],                  # Just carriage return
        ["scope\r"],            # Trailing carriage return
        ["scope\n"],            # Trailing newline
    ]

    for scopes in test_cases:
        print(f"\nInput scopes: {scopes!r}")
        security_scopes = SecurityScopes(scopes=scopes)
        print(f"  scope_str: {security_scopes.scope_str!r}")
        print(f"  scope_str.split(): {security_scopes.scope_str.split()}")

if __name__ == "__main__":
    test_split_behavior()
    test_security_scopes()