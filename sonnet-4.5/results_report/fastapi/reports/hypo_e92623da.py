#!/usr/bin/env python3
"""
Property-based test that discovered the OAuth2 scope parsing bug in FastAPI.

This test verifies that scope parsing follows the OAuth2 specification (RFC 6749),
which requires scopes to be separated by space characters (0x20) only.
"""

from hypothesis import given, strategies as st

@given(st.lists(st.text(min_size=1).filter(lambda x: " " not in x), min_size=1))
def test_oauth2_scope_spec_compliance(scopes_list):
    """
    Test that scope parsing follows OAuth2 RFC 6749 specification.

    According to the spec, scopes must be separated by space characters only,
    not by any whitespace character.
    """
    scope_string = " ".join(scopes_list)

    # How FastAPI currently parses (using split())
    parsed_scopes_with_split = scope_string.split()

    # How OAuth2 spec says it should be parsed (using split(" "))
    parsed_scopes_with_space_split = scope_string.split(" ")

    # These should be equal according to OAuth2 spec
    assert parsed_scopes_with_split == parsed_scopes_with_space_split, \
        f"Scope parsing differs: split()={parsed_scopes_with_split} vs split(' ')={parsed_scopes_with_space_split}"

if __name__ == "__main__":
    # Run the test
    test_oauth2_scope_spec_compliance()