from hypothesis import given, strategies as st, settings
from starlette.middleware.wsgi import build_environ


@given(port=st.integers(min_value=1, max_value=65535))
@settings(max_examples=200)
def test_server_port_is_string_per_pep3333(port):
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

    assert isinstance(environ["SERVER_PORT"], str), \
        f"PEP 3333 requires SERVER_PORT to be a string, got {type(environ['SERVER_PORT'])}"

if __name__ == "__main__":
    test_server_port_is_string_per_pep3333()