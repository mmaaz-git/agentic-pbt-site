"""Test reproduction for HTTPBasic UTF-8 bug report"""
import asyncio
from base64 import b64encode
from hypothesis import given, strategies as st, settings, assume
from fastapi.security.http import HTTPBasic, HTTPBasicCredentials
from starlette.requests import Request


async def create_test_request(authorization_header):
    scope = {
        "type": "http",
        "method": "GET",
        "headers": [[b"authorization", authorization_header.encode()]],
    }
    return Request(scope)


@given(
    st.text(min_size=1).filter(lambda s: ':' not in s and '\x00' not in s),
    st.text().filter(lambda s: '\x00' not in s)
)
@settings(max_examples=200)
def test_http_basic_auth_roundtrip(username, password):
    credentials_str = f"{username}:{password}"

    try:
        encoded = b64encode(credentials_str.encode("utf-8")).decode("ascii")
    except:
        assume(False)

    auth_header = f"Basic {encoded}"

    async def run_test():
        request = await create_test_request(auth_header)
        http_basic = HTTPBasic()
        result = await http_basic(request)

        assert result is not None
        assert isinstance(result, HTTPBasicCredentials)
        assert result.username == username
        assert result.password == password

    asyncio.run(run_test())


async def test_non_ascii_password():
    """Test the specific non-ASCII password case"""
    username = "user"
    password = "päss"

    credentials_str = f"{username}:{password}"
    encoded = b64encode(credentials_str.encode("utf-8")).decode("ascii")
    auth_header = f"Basic {encoded}"

    scope = {
        "type": "http",
        "method": "GET",
        "headers": [[b"authorization", auth_header.encode()]],
    }
    request = Request(scope)

    http_basic = HTTPBasic()
    result = await http_basic(request)

    print(f"Username: {result.username}")
    print(f"Password: {result.password}")
    return result


async def test_specific_failing_input():
    """Test the specific failing input from the bug report"""
    username = '0'
    password = '\x80'

    credentials_str = f"{username}:{password}"
    encoded = b64encode(credentials_str.encode("utf-8")).decode("ascii")
    auth_header = f"Basic {encoded}"

    scope = {
        "type": "http",
        "method": "GET",
        "headers": [[b"authorization", auth_header.encode()]],
    }
    request = Request(scope)

    http_basic = HTTPBasic()
    result = await http_basic(request)

    print(f"Username: {result.username}")
    print(f"Password: {result.password}")
    return result


if __name__ == "__main__":
    # Test with property-based testing
    print("Running property-based test...")
    try:
        test_http_basic_auth_roundtrip()
        print("Property-based test passed!")
    except Exception as e:
        print(f"Property-based test failed: {e}")

    # Test non-ASCII password
    print("\nTesting non-ASCII password (päss)...")
    try:
        result = asyncio.run(test_non_ascii_password())
        print(f"Test passed! Got credentials: {result.username}:{result.password}")
    except Exception as e:
        print(f"Test failed with error: {type(e).__name__}: {e}")

    # Test specific failing input
    print("\nTesting specific failing input (username='0', password='\\x80')...")
    try:
        result = asyncio.run(test_specific_failing_input())
        print(f"Test passed! Got credentials: {result.username}:{result.password}")
    except Exception as e:
        print(f"Test failed with error: {type(e).__name__}: {e}")