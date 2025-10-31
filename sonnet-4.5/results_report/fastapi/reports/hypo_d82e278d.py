import asyncio
from hypothesis import given, strategies as st, settings
from fastapi.security.oauth2 import OAuth2PasswordBearer
from starlette.datastructures import Headers


class MockRequest:
    def __init__(self, headers_dict):
        self.headers = Headers(headers_dict)


@given(st.just("Bearer"))
@settings(max_examples=10)
def test_oauth2_with_scheme_only_should_return_none(scheme_only):
    oauth2 = OAuth2PasswordBearer(tokenUrl="/token", auto_error=False)
    request = MockRequest({"Authorization": scheme_only})
    result = asyncio.run(oauth2(request))
    assert result is None, f"Expected None for Authorization: '{scheme_only}', but got {repr(result)}"


if __name__ == "__main__":
    # Run the property-based test
    test_oauth2_with_scheme_only_should_return_none()