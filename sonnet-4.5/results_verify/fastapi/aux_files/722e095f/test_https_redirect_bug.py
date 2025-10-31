#!/usr/bin/env python3
"""Test to reproduce the HTTPSRedirectMiddleware bug"""

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

    netloc = url.hostname if url.port in (80, 443) else url.netloc
    result_url = url.replace(scheme="https", netloc=netloc)

    assert ":443" in str(result_url), \
        f"Port 443 should be preserved when redirecting http://{hostname}:443 to https, " \
        f"but got {result_url}. Port 443 is non-standard for HTTP."


def reproduce_bug():
    """Reproduce the bug with a concrete example"""
    scope = {
        "type": "http",
        "scheme": "http",
        "server": ("example.com", 443),
        "path": "/test",
        "query_string": b"",
        "headers": []
    }

    url = URL(scope=scope)

    redirect_scheme = "https"
    netloc = url.hostname if url.port in (80, 443) else url.netloc
    result_url = url.replace(scheme=redirect_scheme, netloc=netloc)

    print(f"Original URL: {url}")
    print(f"Redirect URL: {result_url}")
    print(f"Port 443 preserved?: {':443' in str(result_url)}")
    return url, result_url


if __name__ == "__main__":
    # First run the concrete example
    print("=== Concrete Example ===")
    orig, redir = reproduce_bug()

    print("\n=== Running Hypothesis Test ===")
    try:
        test_http_on_port_443_loses_port_on_redirect()
        print("Test passed (unexpected)")
    except AssertionError as e:
        print(f"Test failed as expected: {e}")

    # Test with a specific example
    print("\n=== Testing with hostname='example' ===")
    try:
        test_http_on_port_443_loses_port_on_redirect.hypothesis.inner_test('example')
    except AssertionError as e:
        print(f"Failed: {e}")