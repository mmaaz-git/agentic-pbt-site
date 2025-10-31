from hypothesis import given, settings, strategies as st
from starlette.middleware.trustedhost import TrustedHostMiddleware

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

if __name__ == "__main__":
    test_trustedhost_ipv6_addresses()