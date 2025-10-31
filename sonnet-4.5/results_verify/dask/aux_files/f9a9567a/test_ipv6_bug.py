#!/usr/bin/env python3
"""Test IPv6 address parsing bug in TrustedHostMiddleware"""

# First test the property-based test itself
from hypothesis import given, strategies as st, example


@example(ipv6="::1", port="8000")
@example(ipv6="2001:db8::1", port="8080")
@given(
    st.text(min_size=1, max_size=20),
    st.text(min_size=1, max_size=5, alphabet=st.characters(min_codepoint=48, max_codepoint=57))
)
def test_trustedhost_ipv6_parsing(ipv6, port):
    host_with_port = f"[{ipv6}]:{port}"

    host_extracted = host_with_port.split(":")[0]

    if "[" in host_with_port:
        bracket_close = host_with_port.index("]")
        correct_host = host_with_port[1:bracket_close]

        assert host_extracted == f"[{ipv6}]" or host_extracted == ipv6, (
            f"IPv6 address should be correctly extracted. Got '{host_extracted}' for input '[{ipv6}]:{port}'"
        )


# Run the test with specific examples
print("Testing property-based test with examples...")
try:
    test_trustedhost_ipv6_parsing("::1", "8000")
    print("FAIL: Test passed but should have failed!")
except AssertionError as e:
    print(f"SUCCESS: Test correctly failed with: {e}")

try:
    test_trustedhost_ipv6_parsing("2001:db8::1", "8080")
    print("FAIL: Test passed but should have failed!")
except AssertionError as e:
    print(f"SUCCESS: Test correctly failed with: {e}")