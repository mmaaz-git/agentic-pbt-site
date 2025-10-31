#!/usr/bin/env python3

# Test 1: Run the hypothesis test
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

# Test 2: Specific reproduction case
def test_specific_case():
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/test",
        "query_string": b"",
        "headers": [],
        "server": ("example.com", 8080),
        "http_version": "1.1",
    }

    environ = build_environ(scope, b"")
    print(f"SERVER_PORT: {environ['SERVER_PORT']!r}")
    print(f"Type: {type(environ['SERVER_PORT'])}")
    print(f"Is string? {isinstance(environ['SERVER_PORT'], str)}")

    # Check that it's actually an int (the bug)
    assert isinstance(environ["SERVER_PORT"], int), "Bug confirmed: SERVER_PORT is an int"
    assert environ["SERVER_PORT"] == 8080

if __name__ == "__main__":
    # First run the specific case
    print("Testing specific case...")
    test_specific_case()
    print("\nBug reproduced: SERVER_PORT is an integer, not a string as required by PEP 3333\n")

    # Try the hypothesis test
    print("Running hypothesis test...")
    try:
        test_server_port_is_string_per_pep3333()
        print("Hypothesis test passed (should have failed!)")
    except AssertionError as e:
        print(f"Hypothesis test failed as expected: {e}")