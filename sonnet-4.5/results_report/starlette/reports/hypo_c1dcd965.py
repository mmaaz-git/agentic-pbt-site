import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/starlette_env/lib/python3.13/site-packages')

import pytest
from hypothesis import given, settings, example
import hypothesis.strategies as st


@given(st.sampled_from([
    "[::1]:8000",
    "[2001:db8::1]:443",
    "[fe80::1]:80",
    "[::ffff:192.0.2.1]:8080"
]))
def test_host_header_ipv6_parsing(host_header):
    """Test that TrustedHostMiddleware correctly parses IPv6 addresses in Host headers."""
    # What the middleware currently does (buggy behavior)
    result = host_header.split(":")[0]

    # What it should do for IPv6 addresses
    if host_header.startswith("[") and "]:" in host_header:
        # IPv6 with port: extract address between brackets
        expected = host_header.split("]:")[0][1:]
    elif host_header.startswith("[") and host_header.endswith("]"):
        # IPv6 without port: extract address between brackets
        expected = host_header[1:-1]
    else:
        # Not IPv6 format
        expected = host_header.split(":")[0]

    # This assertion will fail, demonstrating the bug
    assert result == expected, f"Failed to parse {host_header}: got '{result}', expected '{expected}'"


if __name__ == "__main__":
    # Run the test
    test_host_header_ipv6_parsing()