#!/usr/bin/env python3
"""Test functional behavior of CORS middleware with case-sensitive headers"""

from starlette.middleware.cors import CORSMiddleware
from starlette.datastructures import Headers

def dummy_app(scope, receive, send):
    pass

# Create two middlewares with different case headers
m1 = CORSMiddleware(dummy_app, allow_headers=['X-Custom-Header'], allow_origins=['*'])
m2 = CORSMiddleware(dummy_app, allow_headers=['x-custom-header'], allow_origins=['*'])

print("=== Test Preflight Validation ===")
print("m1.allow_headers (configured with 'X-Custom-Header'):", m1.allow_headers)
print("m2.allow_headers (configured with 'x-custom-header'):", m2.allow_headers)

# Test preflight request validation (line 128-131 in cors.py)
# The validation happens at line 129: if header.strip() not in self.allow_headers:

test_headers = [
    "x-custom-header",
    "X-Custom-Header",
    "X-CUSTOM-HEADER"
]

for test_header in test_headers:
    # Simulate headers from a preflight request
    request_headers_dict = {
        'origin': 'http://example.com',
        'access-control-request-method': 'GET',
        'access-control-request-headers': test_header
    }

    headers = Headers(scope={'headers': [(k.encode(), v.encode()) for k, v in request_headers_dict.items()]})

    print(f"\nTesting requested header: '{test_header}'")

    # Check if it would be allowed by m1 and m2
    # This simulates the check at line 128-131
    requested = test_header.lower()
    print(f"  Lowercased to: '{requested}'")
    print(f"  Would pass m1 validation? {requested in m1.allow_headers}")
    print(f"  Would pass m2 validation? {requested in m2.allow_headers}")

print("\n=== Key Observation ===")
print("The validation at line 129 compares lowercased requested headers against")
print("self.allow_headers which is also lowercased (line 67), so validation works correctly.")
print("\nThe issue is ONLY in the internal state representation, not in the actual CORS validation.")
print("Both middlewares will correctly allow the same headers despite different internal ordering.")