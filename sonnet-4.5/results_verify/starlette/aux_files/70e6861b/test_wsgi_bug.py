#!/usr/bin/env python3
"""Test to reproduce the WSGI SERVER_PORT bug"""

from hypothesis import given, settings
import hypothesis.strategies as st
from starlette.middleware.wsgi import build_environ


# First, let's run the property-based test
@given(st.integers(min_value=1, max_value=65535))
@settings(max_examples=100)
def test_server_port_type(port):
    """Property-based test to check SERVER_PORT type"""
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


# Run the property-based test
print("Running property-based test...")
try:
    test_server_port_type()
    print("Property-based test passed (unexpected!)")
except AssertionError as e:
    print(f"Property-based test failed as expected: {e}")


# Now let's reproduce the specific example
print("\nReproducing specific example...")
scope = {
    "type": "http",
    "method": "GET",
    "path": "/",
    "query_string": b"",
    "headers": [],
    "server": ("example.com", 8000),
    "http_version": "1.1",
}

environ = build_environ(scope, b"")

print(f"SERVER_PORT value: {repr(environ['SERVER_PORT'])}")
print(f"SERVER_PORT type: {type(environ['SERVER_PORT'])}")
print(f"Is SERVER_PORT a string? {isinstance(environ['SERVER_PORT'], str)}")

# Test with different port values
print("\nTesting with various port values:")
for port in [80, 443, 8080, 1, 65535]:
    test_scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "query_string": b"",
        "headers": [],
        "server": ("localhost", port),
        "http_version": "1.1",
    }
    env = build_environ(test_scope, b"")
    print(f"  Port {port}: type={type(env['SERVER_PORT'])}, value={repr(env['SERVER_PORT'])}")