from hypothesis import given, strategies as st, settings
from fastapi.security.oauth2 import SecurityScopes


@given(st.lists(st.text(alphabet=st.characters(blacklist_characters=" \t\n\r"), min_size=1), min_size=1))
@settings(max_examples=100)
def test_scopes_roundtrip(scopes_list):
    security_scopes = SecurityScopes(scopes=scopes_list)

    assert security_scopes.scopes == scopes_list

    reconstructed_scopes = security_scopes.scope_str.split()
    assert reconstructed_scopes == scopes_list, \
        f"Failed roundtrip: {scopes_list} -> '{security_scopes.scope_str}' -> {reconstructed_scopes}"

if __name__ == "__main__":
    # Run the test
    test_scopes_roundtrip()