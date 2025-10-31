#!/usr/bin/env python3
"""Real-world impact demonstration from bug report"""

from fastapi.security import OAuth2PasswordBearer
import asyncio
from starlette.datastructures import Headers

class MockRequest:
    def __init__(self, auth_header):
        self.headers = Headers({"authorization": auth_header})

oauth2 = OAuth2PasswordBearer(tokenUrl="/token", auto_error=False)

async def test_impact():
    token1 = await oauth2(MockRequest("Bearer token123"))
    token2 = await oauth2(MockRequest("Bearer  token123"))  # Two spaces

    print(f"Single space: {repr(token1)}")
    print(f"Double space: {repr(token2)}")
    print(f"Tokens match: {token1 == token2}")

    if token1 != token2:
        print(f"\n⚠️  Authentication would fail!")
        print(f"   Token with single space: {repr(token1)}")
        print(f"   Token with double space: {repr(token2)}")
        print(f"   The leading space makes them different tokens")

# Run the test
asyncio.run(test_impact())