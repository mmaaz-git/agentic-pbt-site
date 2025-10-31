import asyncio
import base64
from hypothesis import given, strategies as st, assume
from fastapi.security import HTTPBasic
from starlette.requests import Request
from starlette.datastructures import Headers


@given(
    username=st.text(min_size=1, max_size=50).filter(lambda x: ':' not in x),
    password=st.text(min_size=0, max_size=50)
)
def test_httpbasic_utf8_support(username, password):
    """
    Property: HTTPBasic should support UTF-8 encoded credentials per RFC 7617.
    This test fails when credentials contain non-ASCII UTF-8 characters.
    """
    assume(any(ord(c) > 127 for c in username + password))

    credentials_str = f"{username}:{password}"
    encoded = base64.b64encode(credentials_str.encode('utf-8')).decode('ascii')
    auth_header = f"Basic {encoded}"

    request = Request({
        'type': 'http',
        'headers': Headers({'authorization': auth_header}).raw,
    })

    security = HTTPBasic()

    async def decode_credentials():
        result = await security(request)
        assert result.username == username
        assert result.password == password

    try:
        asyncio.run(decode_credentials())
    except Exception as e:
        print(f"Failed with username='{username}', password='{password}'")
        print(f"Error: {e}")
        return False
    return True


def test_specific_example():
    username = "user"
    password = "pÄssw0rd"

    credentials_str = f"{username}:{password}"
    encoded = base64.b64encode(credentials_str.encode('utf-8')).decode('ascii')
    auth_header = f"Basic {encoded}"

    request = Request({
        'type': 'http',
        'headers': Headers({'authorization': auth_header}).raw,
    })

    security = HTTPBasic()

    async def decode_credentials():
        result = await security(request)
        assert result.username == username
        assert result.password == password

    try:
        asyncio.run(decode_credentials())
        print(f"SUCCESS: Test passed with username='{username}', password='{password}'")
        return True
    except Exception as e:
        print(f"FAILED with username='{username}', password='{password}'")
        print(f"Error: {e}")
        return False


if __name__ == "__main__":
    # Test the specific example from the bug report
    result = test_specific_example()
    print(f"Test result for specific example: {result}")