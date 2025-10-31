#!/usr/bin/env python3
"""Hypothesis property test for IPv6 parsing bug"""

from starlette.datastructures import Headers

def test_ipv6_host_parsing(port):
    """Test that IPv6 addresses with ports are parsed correctly"""
    host_with_port = f"[::1]:{port}"
    headers = Headers({"host": host_with_port})

    # This is what the current implementation does
    host = headers.get("host", "").split(":")[0]

    # Check if it's correct
    expected = "[::1]"
    if host != expected:
        print(f"FAILED with port {port}:")
        print(f"  Input: '{host_with_port}'")
        print(f"  Expected: '{expected}'")
        print(f"  Got: '{host}'")
        return False
    else:
        print(f"PASSED with port {port}: correctly extracted '{host}' from '{host_with_port}'")
        return True

# Test with sample ports
print("Testing IPv6 host parsing with different ports...")
test_ipv6_host_parsing(8080)
test_ipv6_host_parsing(443)
test_ipv6_host_parsing(3000)
test_ipv6_host_parsing(80)
test_ipv6_host_parsing(65535)