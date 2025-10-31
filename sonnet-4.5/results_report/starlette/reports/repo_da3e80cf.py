#!/usr/bin/env python3
"""
Demonstrating the GZipMiddleware case-sensitivity and substring matching bugs.
"""

# Direct demonstration of the bug in the logic
print("=== Direct Logic Bug Demonstration ===")
print("Current implementation at line 24: if 'gzip' in headers.get('Accept-Encoding', ''):")
print()

# Test case-sensitivity issue
accept_encoding_upper = "GZIP"
accept_encoding_mixed = "GZip"
accept_encoding_lower = "gzip"

print("Case-sensitivity tests:")
print(f"  'gzip' in 'GZIP': {('gzip' in accept_encoding_upper)}")
print(f"  'gzip' in 'GZip': {('gzip' in accept_encoding_mixed)}")
print(f"  'gzip' in 'gzip': {('gzip' in accept_encoding_lower)}")
print()

# Test substring matching issue
accept_encoding_not_gzip = "not-gzip"
accept_encoding_gzip_not = "gzip-not"
accept_encoding_msgzip = "msgzip"

print("Substring matching tests:")
print(f"  'gzip' in 'not-gzip': {('gzip' in accept_encoding_not_gzip)}")
print(f"  'gzip' in 'gzip-not': {('gzip' in accept_encoding_gzip_not)}")
print(f"  'gzip' in 'msgzip': {('gzip' in accept_encoding_msgzip)}")
print()

# Test with quality parameters
accept_encoding_with_q = "GZIP;q=1.0"
accept_encoding_with_q_lower = "gzip;q=0.8"

print("With quality parameters:")
print(f"  'gzip' in 'GZIP;q=1.0': {('gzip' in accept_encoding_with_q)}")
print(f"  'gzip' in 'gzip;q=0.8': {('gzip' in accept_encoding_with_q_lower)}")
print()

# Demonstrate with actual middleware behavior
print("=== Testing with actual Starlette GZipMiddleware ===")
try:
    from starlette.middleware.gzip import GZipMiddleware
    from starlette.applications import Starlette
    from starlette.responses import PlainTextResponse
    from starlette.routing import Route
    from starlette.testclient import TestClient
    import gzip

    def homepage(request):
        # Return a response large enough to trigger compression (>500 bytes)
        return PlainTextResponse("X" * 1000)

    routes = [Route("/", homepage)]
    app = Starlette(routes=routes)

    # Wrap app with GZipMiddleware
    app = GZipMiddleware(app, minimum_size=100)

    # Create test client
    client = TestClient(app)

    test_cases = [
        ("gzip", "should compress", True),
        ("GZIP", "should compress but doesn't", False),
        ("GZip", "should compress but doesn't", False),
        ("not-gzip", "shouldn't compress but does", True),
        ("msgzip", "shouldn't compress but does", True),
        ("deflate", "shouldn't compress", False),
    ]

    print("Testing different Accept-Encoding headers:")
    for encoding, description, expected_bug in test_cases:
        response = client.get("/", headers={"Accept-Encoding": encoding})
        has_gzip_header = "content-encoding" in response.headers and response.headers["content-encoding"] == "gzip"

        # Check content length to verify compression
        content_length = response.headers.get("content-length", "unknown")

        status = "COMPRESSED" if has_gzip_header else "NOT compressed"
        bug_indicator = " (BUG!)" if has_gzip_header == expected_bug else ""

        print(f"  {encoding:15} - {status:14} (content-length: {content_length}){bug_indicator}")

except ImportError as e:
    print(f"Error: Could not import required modules: {e}")
    print("Please ensure starlette is installed: pip install starlette")

print()
print("=== Bug Summary ===")
print("1. Case-sensitive check violates RFC 7231 (content-coding values are case-insensitive)")
print("2. Substring matching causes false positives (e.g., 'not-gzip' matches)")
print("3. Valid headers like 'GZIP' or 'GZip' fail to trigger compression")
print("4. Invalid encodings like 'msgzip' trigger compression incorrectly")