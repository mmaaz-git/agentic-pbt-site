import asyncio
from fastapi.security.oauth2 import OAuth2PasswordBearer
from starlette.datastructures import Headers


class MockRequest:
    def __init__(self, headers_dict):
        self.headers = Headers(headers_dict)


# Test 1: Reproduce the basic bug
print("=== Basic Bug Reproduction ===")
oauth2 = OAuth2PasswordBearer(tokenUrl="/token", auto_error=False)

request_missing = MockRequest({})
result_missing = asyncio.run(oauth2(request_missing))
print(f"Missing header: {repr(result_missing)}")

request_empty_token = MockRequest({"Authorization": "Bearer"})
result_empty_token = asyncio.run(oauth2(request_empty_token))
print(f"Empty token: {repr(result_empty_token)}")

print(f"Inconsistent: {result_missing} != {result_empty_token}")
print()

# Test 2: Test with valid token
print("=== Test with valid token ===")
request_valid = MockRequest({"Authorization": "Bearer valid_token_123"})
result_valid = asyncio.run(oauth2(request_valid))
print(f"Valid token result: {repr(result_valid)}")
print()

# Test 3: Test with whitespace after Bearer
print("=== Test with whitespace after Bearer ===")
request_space = MockRequest({"Authorization": "Bearer "})
result_space = asyncio.run(oauth2(request_space))
print(f"Bearer with space: {repr(result_space)}")
print()

# Test 4: Test with multiple spaces
print("=== Test with multiple spaces ===")
request_spaces = MockRequest({"Authorization": "Bearer   "})
result_spaces = asyncio.run(oauth2(request_spaces))
print(f"Bearer with spaces: {repr(result_spaces)}")
print()

# Test 5: Property-based test from the bug report
print("=== Property-based test ===")
from hypothesis import given, strategies as st, settings

@given(st.just("Bearer"))
@settings(max_examples=10)
def test_oauth2_with_scheme_only_should_return_none(scheme_only):
    oauth2 = OAuth2PasswordBearer(tokenUrl="/token", auto_error=False)
    request = MockRequest({"Authorization": scheme_only})
    result = asyncio.run(oauth2(request))
    print(f"Test input: {repr(scheme_only)}, Result: {repr(result)}")
    assert result is None, f"Expected None but got {repr(result)}"

try:
    test_oauth2_with_scheme_only_should_return_none()
    print("Property-based test PASSED")
except AssertionError as e:
    print(f"Property-based test FAILED: {e}")