from hypothesis import given, strategies as st
from fastapi.security.api_key import APIKeyBase


@given(whitespace=st.text(alphabet=' \t\r\n', min_size=1, max_size=10))
def test_apikey_rejects_whitespace_only(whitespace):
    result = APIKeyBase.check_api_key(whitespace, auto_error=False)

    assert result is None, (
        f"Whitespace-only API key {repr(whitespace)} should be rejected, "
        f"but got: {repr(result)}"
    )

if __name__ == "__main__":
    test_apikey_rejects_whitespace_only()