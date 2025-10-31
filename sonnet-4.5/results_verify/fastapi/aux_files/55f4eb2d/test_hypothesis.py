"""Run hypothesis test to find failing cases"""
from hypothesis import given, strategies as st, settings
from fastapi.security import SecurityScopes

@given(st.lists(st.text(min_size=1), min_size=1, max_size=5))
@settings(max_examples=100)
def test_security_scopes_scope_str_round_trip(scopes_list):
    scopes = SecurityScopes(scopes=scopes_list)
    reconstructed = scopes.scope_str.split()
    if reconstructed != scopes.scopes:
        print(f"Failed on input: {repr(scopes_list)}")
        print(f"  Original: {scopes.scopes}")
        print(f"  scope_str: {repr(scopes.scope_str)}")
        print(f"  Reconstructed: {reconstructed}")
        raise AssertionError(f"Round-trip failed for {repr(scopes_list)}")

if __name__ == "__main__":
    print("Running hypothesis test to find failing cases...")
    try:
        test_security_scopes_scope_str_round_trip()
        print("All tests passed!")
    except Exception as e:
        print(f"Tests failed with error: {e}")