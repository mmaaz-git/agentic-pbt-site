from hypothesis import given, settings
import hypothesis.strategies as st
from starlette.middleware.wsgi import build_environ


@given(st.integers(min_value=1, max_value=65535))
@settings(max_examples=100)
def test_server_port_type(port):
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "query_string": b"",
        "headers": [],
        "server": ("example.com", port),
        "http_version": "1.1",
    }

    environ = build_environ(scope, b"")

    assert isinstance(environ['SERVER_PORT'], str), f"SERVER_PORT should be str, got {type(environ['SERVER_PORT'])}"


if __name__ == "__main__":
    # Run the test
    test_server_port_type()