import asyncio
from fastapi.security.oauth2 import OAuth2PasswordBearer
from starlette.datastructures import Headers


class MockRequest:
    def __init__(self, headers_dict):
        self.headers = Headers(headers_dict)


oauth2 = OAuth2PasswordBearer(tokenUrl="/token", auto_error=False)

# Test 1: Missing Authorization header
request_missing = MockRequest({})
result_missing = asyncio.run(oauth2(request_missing))
print(f"Missing header: {repr(result_missing)}")

# Test 2: Authorization header with "Bearer" but no token
request_empty_token = MockRequest({"Authorization": "Bearer"})
result_empty_token = asyncio.run(oauth2(request_empty_token))
print(f"Empty token (Bearer only): {repr(result_empty_token)}")

# Test 3: Authorization header with "Bearer " (with trailing space)
request_empty_token_space = MockRequest({"Authorization": "Bearer "})
result_empty_token_space = asyncio.run(oauth2(request_empty_token_space))
print(f"Empty token (Bearer with space): {repr(result_empty_token_space)}")

# Test 4: Authorization header with valid token
request_valid_token = MockRequest({"Authorization": "Bearer token123"})
result_valid_token = asyncio.run(oauth2(request_valid_token))
print(f"Valid token: {repr(result_valid_token)}")

# Show the inconsistency
print(f"\nInconsistency detected:")
print(f"Missing header returns: {repr(result_missing)}")
print(f"'Bearer' without token returns: {repr(result_empty_token)}")
print(f"These should be the same (both None) but are different: {result_missing != result_empty_token}")