#!/usr/bin/env python3
"""Test the reported IPv6 bug in TrustedHostMiddleware"""

import asyncio
import sys
from hypothesis import given, strategies as st

# Test 1: The property-based test from the bug report
print("Test 1: Property-based test showing parsing issue")
print("=" * 60)

ipv6_addresses = st.sampled_from([
    "::1",
    "2001:db8::1",
    "fe80::1",
    "::ffff:192.0.2.1",
])

@given(ipv6_addresses, st.integers(min_value=1, max_value=65535))
def test_host_header_with_ipv6_and_port(ipv6, port):
    host_header_value = f"[{ipv6}]:{port}"
    host = host_header_value.split(":")[0]
    expected_host_without_port = f"[{ipv6}]"

    assert host == expected_host_without_port, \
        f"IPv6 host parsing is broken: {host_header_value} -> {host}"

# Run the property test
try:
    test_host_header_with_ipv6_and_port()
    print("Property test passed (unexpected!)")
except AssertionError as e:
    print(f"Property test failed as expected: {e}")

# Manual test with specific example
print("\nManual test with [::1]:8000")
host_header_value = "[::1]:8000"
host = host_header_value.split(":")[0]
expected = "[::1]"
print(f"Input: {host_header_value}")
print(f"Expected: {expected}")
print(f"Actual: {host}")
print(f"Match: {host == expected}")

print("\n" + "=" * 60)
print("Test 2: Starlette TrustedHostMiddleware behavior")
print("=" * 60)

# Test 2: The actual middleware test from the bug report
from starlette.middleware.trustedhost import TrustedHostMiddleware

async def dummy_app(scope, receive, send):
    from starlette.responses import PlainTextResponse
    response = PlainTextResponse("OK")
    await response(scope, receive, send)

async def dummy_receive():
    return {"type": "http.request", "body": b"", "more_body": False}

async def test_middleware():
    messages_sent = []

    async def capture_send(message):
        messages_sent.append(message)

    middleware = TrustedHostMiddleware(
        app=dummy_app,
        allowed_hosts=["[::1]"],
    )

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "query_string": b"",
        "headers": [(b"host", b"[::1]:8000")],
    }

    await middleware(scope, dummy_receive, capture_send)

    print(f"Middleware with allowed_hosts=['[::1]'] and Host: [::1]:8000")
    print(f"Response status: {messages_sent[0].get('status', 'N/A')}")
    print(f"Full response: {messages_sent[0]}")

    if messages_sent[0].get('status') == 400:
        print("Result: Request REJECTED (bug confirmed)")
    else:
        print("Result: Request ACCEPTED")

    # Test without port
    messages_sent = []

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "query_string": b"",
        "headers": [(b"host", b"[::1]")],
    }

    await middleware(scope, dummy_receive, capture_send)

    print(f"\nMiddleware with allowed_hosts=['[::1]'] and Host: [::1] (no port)")
    print(f"Response status: {messages_sent[0].get('status', 'N/A')}")

    if messages_sent[0].get('status') == 400:
        print("Result: Request REJECTED")
    else:
        print("Result: Request ACCEPTED")

    # Test with IPv4 for comparison
    middleware = TrustedHostMiddleware(
        app=dummy_app,
        allowed_hosts=["127.0.0.1"],
    )

    messages_sent = []
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "query_string": b"",
        "headers": [(b"host", b"127.0.0.1:8000")],
    }

    await middleware(scope, dummy_receive, capture_send)

    print(f"\nComparison: Middleware with allowed_hosts=['127.0.0.1'] and Host: 127.0.0.1:8000")
    print(f"Response status: {messages_sent[0].get('status', 'N/A')}")

    if messages_sent[0].get('status') == 400:
        print("Result: Request REJECTED")
    else:
        print("Result: Request ACCEPTED (IPv4 works correctly)")

# Run the async test
asyncio.run(test_middleware())

print("\n" + "=" * 60)
print("Test 3: Testing the actual parsing logic")
print("=" * 60)

# Test what the middleware actually does
from starlette.datastructures import Headers

test_cases = [
    ("[::1]:8000", "[::1]"),
    ("[2001:db8::1]:80", "[2001:db8::1]"),
    ("[fe80::1]:443", "[fe80::1]"),
    ("localhost:8000", "localhost"),
    ("example.com:80", "example.com"),
    ("127.0.0.1:8000", "127.0.0.1"),
]

print("Testing how split(':')[0] works on different hosts:")
for input_host, expected in test_cases:
    actual = input_host.split(":")[0]
    status = "✓" if actual == expected else "✗"
    print(f"{status} {input_host} -> {actual} (expected: {expected})")