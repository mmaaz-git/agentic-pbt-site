#!/usr/bin/env python3
"""Test the proposed fix"""

def is_same_domain_fixed(host, pattern):
    """
    Fixed version that lowercases both host and pattern.
    """
    if not pattern:
        return False

    host = host.lower()  # The fix: lowercase the host as well
    pattern = pattern.lower()
    return (
        pattern[0] == "."
        and (host.endswith(pattern) or host == pattern[1:])
        or pattern == host
    )

# Run the same tests with the fixed version
print("Testing with fixed version:")
print()

print("Test 1 - Hypothesis test case:")
print(f"is_same_domain_fixed('A', 'A') = {is_same_domain_fixed('A', 'A')}")
print()

print("Test 2 - Examples from bug report:")
print(f"is_same_domain_fixed('EXAMPLE.COM', 'EXAMPLE.COM') = {is_same_domain_fixed('EXAMPLE.COM', 'EXAMPLE.COM')}")
print(f"is_same_domain_fixed('Example.COM', 'example.com') = {is_same_domain_fixed('Example.COM', 'example.com')}")
print(f"is_same_domain_fixed('FOO.EXAMPLE.COM', '.example.com') = {is_same_domain_fixed('FOO.EXAMPLE.COM', '.example.com')}")
print()

print("Test 3 - Additional tests:")
print(f"is_same_domain_fixed('example.com', 'EXAMPLE.COM') = {is_same_domain_fixed('example.com', 'EXAMPLE.COM')}")
print(f"is_same_domain_fixed('example.com', 'example.com') = {is_same_domain_fixed('example.com', 'example.com')}")
print(f"is_same_domain_fixed('EXAMPLE.COM', 'example.com') = {is_same_domain_fixed('EXAMPLE.COM', 'example.com')}")
print()

print("Test 4 - Wildcard patterns:")
print(f"is_same_domain_fixed('foo.example.com', '.EXAMPLE.COM') = {is_same_domain_fixed('foo.example.com', '.EXAMPLE.COM')}")
print(f"is_same_domain_fixed('FOO.EXAMPLE.COM', '.EXAMPLE.COM') = {is_same_domain_fixed('FOO.EXAMPLE.COM', '.EXAMPLE.COM')}")
print(f"is_same_domain_fixed('foo.example.com', '.example.com') = {is_same_domain_fixed('foo.example.com', '.example.com')}")

# Run hypothesis test with fixed version
print("\nRunning hypothesis test with fixed version:")
from hypothesis import given, strategies as st

@given(st.text(min_size=1))
def test_fixed_exact_match(domain):
    result = is_same_domain_fixed(domain, domain)
    assert result is True, f"Failed for domain='{domain}'"

try:
    test_fixed_exact_match()
    print("Hypothesis test PASSED with fixed version!")
except AssertionError as e:
    print(f"Hypothesis test failed: {e}")