#!/usr/bin/env python3
"""Test the actual TrustedHostMiddleware behavior with IPv6 addresses"""

import asyncio
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.datastructures import Headers
from starlette.requests import Request
from starlette.responses import Response
from starlette.testclient import TestClient
from starlette.applications import Starlette

# First, test the parsing issue directly
def test_parsing_issue():
    print("Testing parsing issue directly:")
    print("-" * 50)

    host_header = "[2001:db8::1]:8080"
    parsed_host = host_header.split(":")[0]

    print(f"Host header: {host_header}")
    print(f"Parsed (buggy): {parsed_host}")
    print(f"Expected: [2001:db8::1] or 2001:db8::1")
    print(f"Parsing correct? {parsed_host in ['[2001:db8::1]', '2001:db8::1']}")

    return parsed_host

# Test the actual middleware behavior
async def test_middleware_behavior():
    print("\n\nTesting actual middleware behavior:")
    print("-" * 50)

    # Create a simple app
    async def app(scope, receive, send):
        response = Response("OK")
        await response(scope, receive, send)

    # Create middleware with IPv6 addresses in allowed hosts
    middleware = TrustedHostMiddleware(
        app,
        allowed_hosts=["[2001:db8::1]", "2001:db8::1", "localhost"]
    )

    # Create a test scope with IPv6 host header
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": [
            (b"host", b"[2001:db8::1]:8080"),
        ],
    }

    # Store responses here
    responses = []

    async def receive():
        return {"type": "http.request", "body": b""}

    async def send(message):
        responses.append(message)

    # Test the middleware
    await middleware(scope, receive, send)

    # Check if we got a proper response or an error
    print(f"Number of response messages: {len(responses)}")
    for i, resp in enumerate(responses):
        print(f"Response {i}: {resp}")

    # Check if it's a 400 error (which would indicate the bug)
    if responses and responses[0].get("status") == 400:
        print("\n✗ CONFIRMED BUG: Middleware rejected valid IPv6 address with 400 status")
        return False
    elif responses and responses[0].get("status") == 200:
        print("\n✓ No bug: Middleware accepted the IPv6 address")
        return True
    else:
        print("\n? Unexpected response")
        return None

# Test with Starlette test client for more complete testing
def test_with_client():
    print("\n\nTesting with Starlette TestClient:")
    print("-" * 50)

    app = Starlette()

    @app.route("/")
    async def homepage(request):
        return Response("OK")

    # Add the middleware
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["[2001:db8::1]", "2001:db8::1", "localhost", "testserver"]
    )

    with TestClient(app) as client:
        # Test with normal host (should work)
        response = client.get("/", headers={"Host": "localhost"})
        print(f"Request to localhost: {response.status_code}")

        # Test with IPv6 address with port (this is where the bug occurs)
        response = client.get("/", headers={"Host": "[2001:db8::1]:8080"})
        print(f"Request to [2001:db8::1]:8080: {response.status_code}")

        # Test with IPv6 address without port
        response = client.get("/", headers={"Host": "[2001:db8::1]"})
        print(f"Request to [2001:db8::1]: {response.status_code}")

        return response.status_code == 200

if __name__ == "__main__":
    # Test parsing issue
    parsed = test_parsing_issue()

    # Test middleware behavior
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(test_middleware_behavior())

    # Test with client
    client_result = test_with_client()