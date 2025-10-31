import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/starlette_env/lib/python3.13/site-packages')

print("Testing the property-based test from the bug report:")
print("=" * 60)

# First, test the property-based test itself
from hypothesis import given, settings, example
import hypothesis.strategies as st

@given(st.sampled_from([
    "[::1]:8000",
    "[2001:db8::1]:443",
    "[fe80::1]:80",
    "[::ffff:192.0.2.1]:8080"
]))
@settings(max_examples=10)
def test_host_header_ipv6_parsing(host_header):
    result = host_header.split(":")[0]

    if ']:' in host_header:
        expected = host_header.split(']:')[0][1:]
        print(f"Testing: {host_header}")
        print(f"  Got: {result}")
        print(f"  Expected: {expected}")
        assert result == expected, f"Failed to parse {host_header}: got {result}, expected {expected}"

# Run the hypothesis test
try:
    test_host_header_ipv6_parsing()
    print("Property-based test passed")
except AssertionError as e:
    print(f"Property-based test failed: {e}")

print("\n" + "=" * 60)
print("Testing the reproduction example:")
print("=" * 60)

# Now test the specific example
host_header = "[::1]:8000"
extracted_host = host_header.split(":")[0]

print(f"Host header: {host_header}")
print(f"Extracted host: {extracted_host}")
print(f"Expected: ::1")
print(f"Got: {extracted_host}")

try:
    assert extracted_host == "::1"
    print("Assertion passed")
except AssertionError:
    print("Assertion failed: extracted_host != '::1'")

print("\n" + "=" * 60)
print("Testing additional IPv6 examples:")
print("=" * 60)

test_cases = [
    ("[::1]:8000", "::1"),
    ("[2001:db8::1]:443", "2001:db8::1"),
    ("[fe80::1]:80", "fe80::1"),
    ("[::ffff:192.0.2.1]:8080", "::ffff:192.0.2.1"),
    ("[::1]", "::1"),  # Without port
]

for host_header, expected in test_cases:
    result = host_header.split(":")[0]
    print(f"Input: {host_header}")
    print(f"  split(':')[0] gives: '{result}'")
    print(f"  Expected: '{expected}'")
    print(f"  Match: {result == expected}")
    print()