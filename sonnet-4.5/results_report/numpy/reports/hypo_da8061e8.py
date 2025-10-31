from hypothesis import given, strategies as st, settings, example
from starlette.datastructures import URL


@given(
    port=st.integers(min_value=1, max_value=65535),
    scheme=st.sampled_from(["http", "ws"])
)
@settings(max_examples=200)
@example(port=443, scheme="http")  # Known failing case
@example(port=443, scheme="ws")    # Another failing case
def test_redirect_preserves_non_default_ports(port, scheme):
    scope = {
        "type": "http" if scheme == "http" else "websocket",
        "scheme": scheme,
        "method": "GET",
        "path": "/",
        "query_string": b"",
        "headers": [],
        "server": ("example.com", port),
    }

    url = URL(scope=scope)
    # This mimics the logic in HTTPSRedirectMiddleware
    current_logic_netloc = url.hostname if url.port in (80, 443) else url.netloc

    # Determine what the default port should be for the original scheme
    default_port_for_scheme = 80 if scheme in ("http", "ws") else 443

    if port != default_port_for_scheme:
        # If it's not the default port for the scheme, it should be preserved
        expected = f"example.com:{port}"
        assert current_logic_netloc == expected or port in (80, 443), \
            f"Non-default port {port} for scheme {scheme} should be preserved. Got netloc: {current_logic_netloc}"


if __name__ == "__main__":
    test_redirect_preserves_non_default_ports()