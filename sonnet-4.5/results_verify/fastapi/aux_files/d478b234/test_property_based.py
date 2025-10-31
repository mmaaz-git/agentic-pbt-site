import pytest
from hypothesis import given, strategies as st, assume
from fastapi.security.utils import get_authorization_scheme_param


@given(
    st.text(min_size=0, max_size=5, alphabet=" \t"),
    st.text(min_size=1).filter(lambda s: not s[0].isspace() and " " not in s and "\t" not in s),
    st.text()
)
def test_leading_whitespace_should_be_ignored(leading_ws, scheme, credentials):
    assume(credentials.strip() == credentials or credentials == "")

    authorization_header = f"{leading_ws}{scheme} {credentials}"

    parsed_scheme, parsed_credentials = get_authorization_scheme_param(authorization_header)

    expected_scheme = scheme
    expected_credentials = credentials

    if leading_ws:
        assert parsed_scheme != expected_scheme or parsed_credentials != expected_credentials, \
            f"Bug: Leading whitespace {leading_ws!r} should be stripped but isn't"

# Test the specific failing case
def test_specific_case():
    authorization_header = " Bearer token123"
    parsed_scheme, parsed_credentials = get_authorization_scheme_param(authorization_header)
    print(f"Input: {authorization_header!r}")
    print(f"Parsed scheme: {parsed_scheme!r}")
    print(f"Parsed credentials: {parsed_credentials!r}")

    # According to the bug report, this should fail
    assert parsed_scheme == "" and parsed_credentials == "Bearer token123"

if __name__ == "__main__":
    test_specific_case()