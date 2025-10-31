#!/usr/bin/env python3
from hypothesis import given, strategies as st
import pytest


def extract_host_current_implementation(host_header):
    """This is the exact implementation from line 40 of trustedhost.py"""
    return host_header.split(":")[0]


@st.composite
def ipv6_addresses(draw):
    segments = draw(st.lists(
        st.text(min_size=1, max_size=4, alphabet='0123456789abcdef'),
        min_size=2,
        max_size=8
    ))
    return "[" + ":".join(segments) + "]"


@given(ipv6_addresses())
def test_ipv6_host_extraction_bug(ipv6_addr):
    extracted = extract_host_current_implementation(ipv6_addr)

    if extracted != ipv6_addr:
        pytest.fail(
            f"IPv6 address parsing failed: "
            f"Input: '{ipv6_addr}' -> Extracted: '{extracted}' (Expected: '{ipv6_addr}')"
        )


if __name__ == "__main__":
    test_ipv6_host_extraction_bug()