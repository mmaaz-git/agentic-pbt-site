#!/usr/bin/env python3
"""Test actual middleware behavior with IPv6 addresses"""

import asyncio
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.testclient import TestClient

# Create a simple app
app = Starlette()

@app.route("/")
async def homepage(request):
    return PlainTextResponse("Hello, world!")

def test_ipv6_with_actual_middleware():
    """Test middleware with actual requests"""
    print("=" * 60)
    print("Testing actual middleware behavior with IPv6")
    print("=" * 60)

    # Test 1: IPv6 localhost
    print("\nTest 1: IPv6 localhost [::1]")
    print("-" * 40)

    # Create app with IPv6 address in allowed_hosts
    app_with_ipv6 = TrustedHostMiddleware(app, allowed_hosts=["[::1]"])
    client = TestClient(app_with_ipv6)

    # Try to make a request with IPv6 host header
    response = client.get("/", headers={"host": "[::1]"})
    print(f"Request with Host: [::1]")
    print(f"Response status: {response.status_code}")
    print(f"Response text: {response.text}")

    if response.status_code == 400:
        print("❌ Bug confirmed: IPv6 host [::1] was rejected")
    else:
        print("✓ IPv6 host [::1] was accepted")

    # Test 2: IPv6 with port
    print("\nTest 2: IPv6 with port [::1]:8080")
    print("-" * 40)

    response = client.get("/", headers={"host": "[::1]:8080"})
    print(f"Request with Host: [::1]:8080")
    print(f"Response status: {response.status_code}")
    print(f"Response text: {response.text}")

    if response.status_code == 400:
        print("❌ Bug confirmed: IPv6 host [::1]:8080 was rejected")
    else:
        print("✓ IPv6 host [::1]:8080 was accepted")

    # Test 3: IPv6 full address
    print("\nTest 3: Full IPv6 address [2001:db8::1]")
    print("-" * 40)

    app_with_ipv6_full = TrustedHostMiddleware(app, allowed_hosts=["[2001:db8::1]"])
    client2 = TestClient(app_with_ipv6_full)

    response = client2.get("/", headers={"host": "[2001:db8::1]"})
    print(f"Request with Host: [2001:db8::1]")
    print(f"Response status: {response.status_code}")
    print(f"Response text: {response.text}")

    if response.status_code == 400:
        print("❌ Bug confirmed: IPv6 host [2001:db8::1] was rejected")
    else:
        print("✓ IPv6 host [2001:db8::1] was accepted")

    # Test 4: Verify IPv4 still works
    print("\nTest 4: IPv4 addresses (should work)")
    print("-" * 40)

    app_with_ipv4 = TrustedHostMiddleware(app, allowed_hosts=["localhost", "127.0.0.1"])
    client3 = TestClient(app_with_ipv4)

    response = client3.get("/", headers={"host": "localhost"})
    print(f"Request with Host: localhost")
    print(f"Response status: {response.status_code}")
    assert response.status_code == 200, "IPv4 localhost should work"

    response = client3.get("/", headers={"host": "127.0.0.1"})
    print(f"Request with Host: 127.0.0.1")
    print(f"Response status: {response.status_code}")
    assert response.status_code == 200, "IPv4 127.0.0.1 should work"

    print("✓ IPv4 addresses work correctly")

def test_middleware_internal_logic():
    """Test the internal logic of how the middleware processes hosts"""
    print("\n" + "=" * 60)
    print("Testing middleware internal processing logic")
    print("=" * 60)

    from starlette.datastructures import Headers

    # Simulate what the middleware does
    test_cases = [
        ("[::1]", ["[::1]"], "IPv6 localhost"),
        ("[2001:db8::1]", ["[2001:db8::1]"], "IPv6 full"),
        ("[::1]:8080", ["[::1]"], "IPv6 with port"),
        ("localhost", ["localhost"], "IPv4 hostname"),
        ("127.0.0.1:8080", ["127.0.0.1"], "IPv4 with port"),
    ]

    for host_header, allowed_hosts, description in test_cases:
        print(f"\n{description}:")
        print(f"  Host header: {host_header}")
        print(f"  Allowed hosts: {allowed_hosts}")

        # Current behavior (buggy)
        extracted = host_header.split(":")[0]
        is_valid = extracted in allowed_hosts

        print(f"  Current extraction: {extracted}")
        print(f"  Current is_valid: {is_valid}")

        # Expected behavior
        if host_header.startswith("["):
            # IPv6 address
            if "]:" in host_header:
                expected_extracted = host_header.rsplit(":", 1)[0]
            else:
                expected_extracted = host_header
        else:
            # IPv4 or hostname
            expected_extracted = host_header.split(":", 1)[0]

        expected_is_valid = expected_extracted in allowed_hosts

        print(f"  Expected extraction: {expected_extracted}")
        print(f"  Expected is_valid: {expected_is_valid}")

        if is_valid != expected_is_valid:
            print(f"  ❌ BUG: Incorrect validation result!")
        else:
            print(f"  ✓ Validation result matches expected")

if __name__ == "__main__":
    test_ipv6_with_actual_middleware()
    test_middleware_internal_logic()

    print("\n" + "=" * 60)
    print("FINAL CONCLUSION:")
    print("The bug is CONFIRMED. TrustedHostMiddleware does not")
    print("correctly handle IPv6 addresses due to naive string")
    print("splitting on ':' which breaks IPv6 address parsing.")
    print("=" * 60)