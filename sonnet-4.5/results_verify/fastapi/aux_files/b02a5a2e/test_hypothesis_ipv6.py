#!/usr/bin/env python3
"""Test the Hypothesis property test from the bug report"""

from hypothesis import given, strategies as st

@given(
    st.lists(st.integers(min_value=0, max_value=65535).map(lambda x: hex(x)[2:]), min_size=8, max_size=8),
    st.integers(min_value=1, max_value=65535)
)
def test_trustedhost_ipv6_parsing(segments, port):
    ipv6 = ':'.join(segments)
    host_with_port = f"[{ipv6}]:{port}"

    parsed = host_with_port.split(":")[0]
    expected = f"[{ipv6}]"

    assert parsed == expected or parsed == ipv6, f"Failed: parsed={parsed}, expected={expected} or {ipv6}"

# Test with the specific failing input from the bug report
def test_specific_failure():
    segments = ['2', '0', '0', '1', 'd', 'b', '8', '1']
    port = 8080

    ipv6 = ':'.join(segments)
    host_with_port = f"[{ipv6}]:{port}"

    print(f"IPv6 address: {ipv6}")
    print(f"Host with port: {host_with_port}")

    parsed = host_with_port.split(":")[0]
    expected = f"[{ipv6}]"

    print(f"Parsed (using split(':')[0]): {parsed}")
    print(f"Expected: {expected} or {ipv6}")
    print(f"Test passes? {parsed == expected or parsed == ipv6}")

    return parsed == expected or parsed == ipv6

if __name__ == "__main__":
    # Run the specific failing case
    print("Testing specific failing case:")
    print("-" * 50)
    result = test_specific_failure()
    print(f"\nResult: {'PASS' if result else 'FAIL'}")

    # Try running hypothesis test
    print("\n" + "=" * 50)
    print("Running Hypothesis test (expecting failures):")
    try:
        test_trustedhost_ipv6_parsing()
        print("Hypothesis test completed without finding failures (unexpected!)")
    except AssertionError as e:
        print(f"Hypothesis test found a failure (expected): {e}")