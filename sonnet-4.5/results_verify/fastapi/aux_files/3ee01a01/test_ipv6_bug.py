from hypothesis import given, strategies as st, example
from starlette.datastructures import Headers

@given(st.sampled_from(["::1", "2001:db8::1", "fe80::1"]), st.integers(min_value=1, max_value=65535))
@example("::1", 8080)
def test_ipv6_host_parsing(ipv6_addr, port):
    host_header = f"[{ipv6_addr}]:{port}"

    scope = {
        'type': 'http',
        'headers': [(b'host', host_header.encode())]
    }

    headers = Headers(scope=scope)
    parsed_host = headers.get("host", "").split(":")[0]

    assert parsed_host == ipv6_addr, f"Expected {ipv6_addr}, got {parsed_host}"

if __name__ == "__main__":
    # Run the test with the specific failing example
    ipv6_addr = "::1"
    port = 8080
    host_header = f"[{ipv6_addr}]:{port}"

    scope = {
        'type': 'http',
        'headers': [(b'host', host_header.encode())]
    }

    headers = Headers(scope=scope)
    parsed_host = headers.get("host", "").split(":")[0]

    print(f"Host header: {host_header}")
    print(f"Expected: {ipv6_addr}")
    print(f"Actual: {parsed_host}")

    try:
        assert parsed_host == ipv6_addr, f"Expected {ipv6_addr}, got {parsed_host}"
        print("Test passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")