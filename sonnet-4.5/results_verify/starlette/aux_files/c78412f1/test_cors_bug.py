#!/usr/bin/env python3
"""Test the reported CORS middleware bug"""

from hypothesis import given, strategies as st, settings
from starlette.datastructures import Headers
from starlette.middleware.cors import CORSMiddleware

# First, reproduce the exact bug from the report
def test_exact_reproduction():
    """Reproduce the exact failing case from the bug report"""
    print("Testing exact reproduction case...")

    middleware = CORSMiddleware(
        app=None,
        allow_origins=["*"],
        allow_methods=["GET "]  # Note the trailing space
    )

    request_headers = Headers({
        "origin": "http://example.com",
        "access-control-request-method": "GET"  # No trailing space
    })

    response = middleware.preflight_response(request_headers)

    print(f"Response status code: {response.status_code}")
    print(f"Expected: 200, Got: {response.status_code}")

    if response.status_code != 200:
        print("BUG CONFIRMED: Method with trailing space in config doesn't match method without space in request")
        return False
    else:
        print("No bug: Method matched successfully")
        return True

# Test with property-based testing
@given(
    method=st.sampled_from(["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"]),
    spaces_before=st.integers(min_value=0, max_value=3),
    spaces_after=st.integers(min_value=0, max_value=3)
)
@settings(max_examples=50)
def test_cors_allow_methods_whitespace(method, spaces_before, spaces_after):
    """Property-based test from the bug report"""
    method_with_spaces = " " * spaces_before + method + " " * spaces_after

    middleware = CORSMiddleware(
        app=None,
        allow_origins=["*"],
        allow_methods=[method_with_spaces]
    )

    request_headers = Headers({
        "origin": "http://example.com",
        "access-control-request-method": method
    })

    response = middleware.preflight_response(request_headers)

    assert response.status_code == 200, \
        f"Expected 200 OK but got {response.status_code}. " \
        f"Method '{method_with_spaces}' (with spaces) was allowed in config, " \
        f"but request method '{method}' (without spaces) was rejected."

def test_reverse_case():
    """Test the reverse: clean config, request with spaces"""
    print("\nTesting reverse case (clean config, request with spaces)...")

    middleware = CORSMiddleware(
        app=None,
        allow_origins=["*"],
        allow_methods=["GET"]  # No spaces
    )

    # Test if request with spaces would match
    request_headers = Headers({
        "origin": "http://example.com",
        "access-control-request-method": " GET "  # With spaces
    })

    response = middleware.preflight_response(request_headers)

    print(f"Response status code: {response.status_code}")
    print(f"Clean config 'GET', request ' GET ': status={response.status_code}")

def test_current_behavior():
    """Test various whitespace scenarios to understand current behavior"""
    print("\nTesting current whitespace handling behavior...")

    test_cases = [
        ("GET", "GET", "Exact match"),
        ("GET ", "GET", "Config trailing space"),
        (" GET", "GET", "Config leading space"),
        (" GET ", "GET", "Config both spaces"),
        ("GET", "GET ", "Request trailing space"),
        ("GET", " GET", "Request leading space"),
        ("GET", " GET ", "Request both spaces"),
    ]

    for config_method, request_method, description in test_cases:
        middleware = CORSMiddleware(
            app=None,
            allow_origins=["*"],
            allow_methods=[config_method]
        )

        request_headers = Headers({
            "origin": "http://example.com",
            "access-control-request-method": request_method
        })

        response = middleware.preflight_response(request_headers)
        status = "✓" if response.status_code == 200 else "✗"
        print(f"{status} {description}: config='{config_method}', request='{request_method}' -> {response.status_code}")

if __name__ == "__main__":
    print("=" * 60)
    print("CORS Middleware Whitespace Handling Bug Test")
    print("=" * 60)

    # Run exact reproduction
    test_exact_reproduction()

    # Run reverse case
    test_reverse_case()

    # Run behavior analysis
    test_current_behavior()

    # Run property-based test
    print("\nRunning property-based tests...")
    try:
        test_cors_allow_methods_whitespace()
        print("Property-based tests passed (unexpected!)")
    except AssertionError as e:
        print(f"Property-based test failed as expected: {e}")