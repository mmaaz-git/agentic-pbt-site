import asyncio
from fastapi.security.oauth2 import OAuth2PasswordBearer
from fastapi.security.utils import get_authorization_scheme_param
from starlette.datastructures import Headers

class MockRequest:
    def __init__(self, headers_dict):
        self.headers = Headers(headers_dict)

# Simulate the current logic in OAuth2PasswordBearer.__call__
def simulate_current_logic(authorization_header, auto_error=False):
    """Simulates the current OAuth2PasswordBearer logic"""
    scheme, param = get_authorization_scheme_param(authorization_header)

    # This is the current condition in the source code
    if not authorization_header or scheme.lower() != "bearer":
        if auto_error:
            return "WOULD RAISE HTTPException(401)"
        else:
            return None
    return param

# Test cases to understand current behavior
test_cases = [
    (None, "Missing header"),
    ("", "Empty string header"),
    ("Bearer", "Bearer without token"),
    ("Bearer ", "Bearer with single space"),
    ("Bearer  ", "Bearer with double space"),
    ("Bearer token123", "Bearer with valid token"),
    ("Basic xyz", "Wrong scheme"),
]

print("Current OAuth2PasswordBearer logic simulation:")
print("=" * 60)
for header, description in test_cases:
    result = simulate_current_logic(header, auto_error=False)
    print(f"{description:30} | Header: {repr(header):20} | Result: {repr(result)}")

print("\n" + "=" * 60)
print("\nKey observation:")
print("When header is 'Bearer' (no token), the condition check is:")
print("  scheme='Bearer', param=''")
print("  not authorization_header = False (header exists)")
print("  scheme.lower() != 'bearer' = False (scheme is 'bearer')")
print("  So the condition is False, and it returns param which is ''")
print("\nThis explains why empty string is returned instead of None!")