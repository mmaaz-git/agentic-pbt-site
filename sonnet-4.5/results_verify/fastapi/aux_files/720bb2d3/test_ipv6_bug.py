#!/usr/bin/env python3
"""Test to reproduce the IPv6 parsing bug in TrustedHostMiddleware"""

# Test 1: Simple string splitting demonstration
print("=== Test 1: String splitting demonstration ===")
test_cases = [
    "[::1]",
    "[::1]:8080",
    "[2001:db8::1]",
    "[2001:db8::1]:8080",
    "localhost",
    "localhost:8080",
    "example.com",
    "example.com:443"
]

for host_header in test_cases:
    host = host_header.split(":")[0]
    print(f"Input: '{host_header}'")
    print(f"Extracted host using split(':')[0]: '{host}'")

    # Show what we'd expect for IPv6
    if host_header.startswith("["):
        expected = host_header.split("]")[0] + "]" if "]" in host_header else host_header
        print(f"Expected for IPv6: '{expected}'")
    else:
        print(f"Expected for IPv4/hostname: '{host}'")
    print()

# Test 2: Test with actual middleware
print("\n=== Test 2: Testing with actual TrustedHostMiddleware ===")
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.datastructures import Headers
import asyncio

class MockApp:
    def __init__(self):
        self.called = False

    async def __call__(self, scope, receive, send):
        self.called = True
        await send({
            'type': 'http.response.start',
            'status': 200,
            'headers': [],
        })
        await send({
            'type': 'http.response.body',
            'body': b'OK',
        })

async def test_middleware():
    # Test with IPv6 in allowed hosts
    app = MockApp()
    middleware = TrustedHostMiddleware(
        app,
        allowed_hosts=["[::1]", "[2001:db8::1]", "localhost"]
    )

    test_hosts = [
        "[::1]",
        "[::1]:8080",
        "[2001:db8::1]",
        "[2001:db8::1]:8080",
        "localhost",
        "localhost:8080"
    ]

    for host_header in test_hosts:
        app.called = False

        # Create a mock scope
        scope = {
            "type": "http",
            "headers": [(b"host", host_header.encode())],
            "path": "/",
            "query_string": b"",
            "method": "GET",
        }

        # Mock receive and send
        async def receive():
            return {"type": "http.request", "body": b""}

        responses = []
        async def send(response):
            responses.append(response)

        # Call the middleware
        await middleware(scope, receive, send)

        # Check if the request was allowed
        if responses:
            status = responses[0].get('status', 0)
            if status == 200:
                print(f"Host '{host_header}': ALLOWED (200 OK)")
            elif status == 400:
                print(f"Host '{host_header}': REJECTED (400 Bad Request)")
            else:
                print(f"Host '{host_header}': Status {status}")
        else:
            print(f"Host '{host_header}': No response")

# Run the async test
asyncio.run(test_middleware())

# Test 3: Hypothesis test
print("\n=== Test 3: Hypothesis property-based test ===")
try:
    from hypothesis import given, strategies as st

    @given(st.integers(min_value=1, max_value=65535))
    def test_ipv6_host_parsing(port):
        host_with_port = f"[::1]:{port}"
        headers = Headers({"host": host_with_port})

        # This is what the current implementation does
        host = headers.get("host", "").split(":")[0]

        # Check if it's correct
        try:
            assert host == "[::1]", f"Expected '[::1]' but got '{host}' when parsing '{host_with_port}'"
            return True
        except AssertionError as e:
            print(f"FAILED with port {port}: {e}")
            return False

    # Run the test with a few examples
    print("Running property test with sample ports...")
    test_ipv6_host_parsing(8080)
    test_ipv6_host_parsing(443)
    test_ipv6_host_parsing(3000)

except ImportError:
    print("Hypothesis not installed, skipping property test")