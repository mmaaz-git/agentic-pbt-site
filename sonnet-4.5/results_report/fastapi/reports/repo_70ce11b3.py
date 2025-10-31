#!/usr/bin/env python3
"""Minimal reproduction of IPv6 address parsing bug in TrustedHostMiddleware"""

def extract_host_current_implementation(host_header):
    """This is the exact implementation from line 40 of trustedhost.py"""
    return host_header.split(":")[0]

# Test cases showing the bug
test_cases = [
    "[::1]",
    "[::1]:8000",
    "[2001:db8::1]",
    "[2001:db8::1]:8080",
    "[fe80::1%eth0]",
    "[::ffff:192.0.2.1]",
    "[0:0]",
    "[::]",
]

print("IPv6 Address Parsing Bug in TrustedHostMiddleware")
print("=" * 50)
print()

for host_header in test_cases:
    extracted = extract_host_current_implementation(host_header)
    expected = host_header.split("]")[0] + "]" if host_header.startswith("[") else host_header.split(":")[0]

    print(f"Input:    '{host_header}'")
    print(f"Output:   '{extracted}'")
    print(f"Expected: '{expected}'")
    print(f"FAIL: {extracted != expected}")
    print()