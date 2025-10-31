from hypothesis import given, strategies as st
from starlette.datastructures import URL


@given(hostname=st.text(alphabet=st.characters(min_codepoint=97, max_codepoint=122), min_size=3, max_size=15))
def test_http_on_port_443_loses_port_on_redirect(hostname):
    scope = {
        "type": "http",
        "scheme": "http",
        "server": (hostname, 443),
        "path": "/",
        "query_string": b"",
        "headers": []
    }

    url = URL(scope=scope)

    # This is the exact logic from HTTPSRedirectMiddleware
    netloc = url.hostname if url.port in (80, 443) else url.netloc
    result_url = url.replace(scheme="https", netloc=netloc)

    assert ":443" in str(result_url), \
        f"Port 443 should be preserved when redirecting http://{hostname}:443 to https, " \
        f"but got {result_url}. Port 443 is non-standard for HTTP."


if __name__ == "__main__":
    test_http_on_port_443_loses_port_on_redirect()