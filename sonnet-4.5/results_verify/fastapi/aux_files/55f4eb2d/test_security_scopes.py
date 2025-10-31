"""Test to reproduce the SecurityScopes bug"""
from hypothesis import given, strategies as st
from fastapi.security import SecurityScopes

# Property-based test from bug report
@given(st.lists(st.text(min_size=1)))
def test_security_scopes_scope_str_round_trip(scopes_list):
    scopes = SecurityScopes(scopes=scopes_list)
    reconstructed = scopes.scope_str.split()
    assert reconstructed == scopes.scopes

# Specific reproduction case from bug report
def test_whitespace_scope():
    scopes_list = [' ']
    scopes = SecurityScopes(scopes=scopes_list)

    print(f"Original scopes: {scopes.scopes}")
    print(f"scope_str: {repr(scopes.scope_str)}")
    print(f"Reconstructed: {scopes.scope_str.split()}")

    # These are the assertions from the bug report
    assert scopes.scopes == [' ']
    assert scopes.scope_str == ' '
    assert scopes.scope_str.split() == []

    # The claim is that round-trip doesn't work
    print(f"Round-trip works? {scopes.scope_str.split() == scopes.scopes}")
    return scopes.scope_str.split() == scopes.scopes

# Let's also test other edge cases
def test_empty_string_scope():
    scopes_list = ['']
    scopes = SecurityScopes(scopes=scopes_list)
    print(f"\nEmpty string test:")
    print(f"Original scopes: {scopes.scopes}")
    print(f"scope_str: {repr(scopes.scope_str)}")
    print(f"Reconstructed: {scopes.scope_str.split()}")
    return scopes.scope_str.split() == scopes.scopes

def test_mixed_valid_invalid():
    scopes_list = ['read', ' ', 'write']
    scopes = SecurityScopes(scopes=scopes_list)
    print(f"\nMixed scopes test:")
    print(f"Original scopes: {scopes.scopes}")
    print(f"scope_str: {repr(scopes.scope_str)}")
    print(f"Reconstructed: {scopes.scope_str.split()}")
    return scopes.scope_str.split() == scopes.scopes

if __name__ == "__main__":
    # Run the specific test case
    try:
        result = test_whitespace_scope()
        print(f"\nWhitespace scope round-trip: {result}")
    except AssertionError as e:
        print(f"\nAssertion failed in whitespace test")

    # Run other tests
    print(f"\nEmpty string round-trip: {test_empty_string_scope()}")
    print(f"Mixed scopes round-trip: {test_mixed_valid_invalid()}")

    # Test the specific failing case without hypothesis
    print("\n\nTesting specific failing input [' ']:")
    try:
        scopes = SecurityScopes(scopes=[' '])
        reconstructed = scopes.scope_str.split()
        assert reconstructed == scopes.scopes
        print("Test passed with [' ']")
    except AssertionError:
        print("Test FAILED with [' '] - round-trip property violated")