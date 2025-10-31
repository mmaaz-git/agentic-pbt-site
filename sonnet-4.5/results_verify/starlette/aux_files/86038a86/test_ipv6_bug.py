#!/usr/bin/env python3
"""Test IPv6 handling in TrustedHostMiddleware"""

from hypothesis import given, settings, strategies as st
from starlette.middleware.trustedhost import TrustedHostMiddleware

# First, let's test the hypothesis test from the bug report
@given(st.sampled_from(["[::1]", "[2001:db8::1]", "[fe80::1]"]))
@settings(max_examples=50)
def test_trustedhost_ipv6_addresses(ipv6_address):
    middleware = TrustedHostMiddleware(None, allowed_hosts=[ipv6_address])

    extracted = ipv6_address.split(":")[0]

    is_valid = False
    for pattern in middleware.allowed_hosts:
        if extracted == pattern or (pattern.startswith("*") and extracted.endswith(pattern[1:])):
            is_valid = True
            break

    assert is_valid is True

# Now let's run the reproduction case
def test_reproduction():
    print("=" * 60)
    print("Testing IPv6 address handling in TrustedHostMiddleware")
    print("=" * 60)

    middleware = TrustedHostMiddleware(None, allowed_hosts=["[::1]"])

    host_header = "[::1]"
    extracted_host = host_header.split(":")[0]

    print(f"Host header: {host_header}")
    print(f"Extracted: {extracted_host}")
    print(f"Expected: [::1]")

    is_valid = False
    for pattern in middleware.allowed_hosts:
        if extracted_host == pattern or (pattern.startswith("*") and extracted_host.endswith(pattern[1:])):
            is_valid = True
            break

    print(f"Is valid: {is_valid}")
    print()
    return is_valid

def test_various_ipv6_formats():
    """Test various IPv6 address formats"""
    print("Testing various IPv6 formats:")
    print("-" * 40)

    test_cases = [
        ("[::1]", "[::1]", "IPv6 loopback"),
        ("[::1]:8080", "[::1]", "IPv6 loopback with port"),
        ("[2001:db8::1]", "[2001:db8::1]", "IPv6 address"),
        ("[2001:db8::1]:8000", "[2001:db8::1]", "IPv6 with port"),
        ("[fe80::1]", "[fe80::1]", "IPv6 link-local"),
        ("[::ffff:192.168.1.1]", "[::ffff:192.168.1.1]", "IPv6 mapped IPv4"),
    ]

    for host_header, expected_host, description in test_cases:
        # What the current code does
        extracted_current = host_header.split(":")[0]

        # What it should do (simplified version of the proposed fix)
        if host_header.startswith("["):
            if "]:" in host_header:
                extracted_correct = host_header.rsplit(":", 1)[0]
            else:
                extracted_correct = host_header
        else:
            extracted_correct = host_header.split(":", 1)[0]

        print(f"{description}:")
        print(f"  Input:           {host_header}")
        print(f"  Current extract: {extracted_current}")
        print(f"  Should extract:  {expected_host}")
        print(f"  Fix extracts:    {extracted_correct}")
        print(f"  Match expected:  {extracted_correct == expected_host}")
        print()

def test_ipv4_still_works():
    """Ensure IPv4 addresses still work correctly"""
    print("Testing IPv4 addresses still work:")
    print("-" * 40)

    test_cases = [
        ("localhost", "localhost"),
        ("localhost:8080", "localhost"),
        ("example.com", "example.com"),
        ("example.com:443", "example.com"),
        ("192.168.1.1", "192.168.1.1"),
        ("192.168.1.1:8000", "192.168.1.1"),
    ]

    for host_header, expected_host in test_cases:
        extracted = host_header.split(":")[0]
        print(f"  {host_header:20} -> {extracted:15} (expected: {expected_host})")
        assert extracted == expected_host, f"IPv4 extraction failed for {host_header}"

    print("All IPv4 tests passed!")
    print()

if __name__ == "__main__":
    import sys

    # Test the reproduction case
    try:
        is_valid = test_reproduction()
        if not is_valid:
            print("❌ Bug confirmed: IPv6 addresses are not handled correctly")
        else:
            print("✓ No bug found")
    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")

    print()

    # Test various formats
    test_various_ipv6_formats()

    # Verify IPv4 still works
    test_ipv4_still_works()

    # Run hypothesis test
    print("Running hypothesis test...")
    print("-" * 40)
    try:
        test_trustedhost_ipv6_addresses("[::1]")
        test_trustedhost_ipv6_addresses("[2001:db8::1]")
        test_trustedhost_ipv6_addresses("[fe80::1]")
        print("✓ Hypothesis tests passed (unexpected!)")
    except AssertionError:
        print("❌ Hypothesis test failed (as expected - bug confirmed)")

    print("\n" + "=" * 60)
    print("CONCLUSION: The bug is REAL. IPv6 addresses are not handled correctly.")
    print("=" * 60)