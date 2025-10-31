#!/usr/bin/env python3
"""Bug reproduction test for FastAPI security scope handling"""

from hypothesis import given, strategies as st
from fastapi.security import OAuth2PasswordRequestForm, SecurityScopes

# Test 1: Property-based test for round-trip
@given(st.lists(st.text(min_size=1).filter(lambda x: not x.isspace() and " " not in x)))
def test_oauth2_scope_round_trip(scope_list):
    scope_string = " ".join(scope_list)
    form = OAuth2PasswordRequestForm(
        username="test_user",
        password="test_pass",
        scope=scope_string
    )
    assert form.scopes == scope_list
    reconstructed = " ".join(form.scopes)
    assert reconstructed == scope_string

# Test 2: Direct reproduction with the failing example
def test_failing_example():
    print("\n=== Test with failing example ===")
    scope_list = ['0\r']
    scope_string = " ".join(scope_list)
    form = OAuth2PasswordRequestForm(
        username="test_user",
        password="test_pass",
        scope=scope_string
    )
    print(f"Input scope_list: {scope_list!r}")
    print(f"Input scope_string: {scope_string!r}")
    print(f"Form scopes: {form.scopes!r}")
    print(f"Reconstructed: {' '.join(form.scopes)!r}")
    print(f"Are they equal? scope_list == form.scopes: {scope_list == form.scopes}")
    print(f"Round-trip preserved? {scope_string == ' '.join(form.scopes)}")

# Test 3: Manual reproduction examples from bug report
def test_manual_reproduction():
    print("\n=== Manual reproduction test ===")

    # Example 1: Multiple spaces
    scope_string = "read  write"
    form = OAuth2PasswordRequestForm(
        username="test",
        password="test",
        scope=scope_string
    )
    print(f"Input:  {scope_string!r}")
    print(f"Output: {' '.join(form.scopes)!r}")
    print(f"Scopes list: {form.scopes!r}")

    # Example 2: SecurityScopes with carriage return
    print("\n--- SecurityScopes test ---")
    security_scopes = SecurityScopes(scopes=['\r'])
    print(f"Input scopes:  {['\r']!r}")
    print(f"Scope string:  {security_scopes.scope_str!r}")
    print(f"After split:   {security_scopes.scope_str.split()!r}")

if __name__ == "__main__":
    # Run the manual tests first
    test_failing_example()
    test_manual_reproduction()

    # Try the hypothesis test
    print("\n=== Running property-based test ===")
    try:
        test_oauth2_scope_round_trip()
        print("All property tests passed!")
    except AssertionError as e:
        print(f"Property test failed: {e}")
        # Try with the specific failing example
        try:
            test_oauth2_scope_round_trip(['0\r'])
        except AssertionError as e2:
            print(f"Confirmed failure with ['0\\r']: {e2}")