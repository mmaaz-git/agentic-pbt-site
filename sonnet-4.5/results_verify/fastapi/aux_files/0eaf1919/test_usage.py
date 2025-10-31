#!/usr/bin/env python3
"""Test how the headers are actually used in the middleware."""

from starlette.middleware.cors import CORSMiddleware
from starlette.datastructures import Headers

# Create middleware with the problematic headers
middleware = CORSMiddleware(
    app=lambda scope, receive, send: None,
    allow_headers=['[', 'X-Custom'],
    allow_origins=['http://example.com'],
    allow_methods=['GET', 'POST']
)

print("Testing header validation in preflight_response:")
print("="*50)

# Simulate a preflight request
mock_headers = {
    'origin': 'http://example.com',
    'access-control-request-method': 'POST',
    'access-control-request-headers': 'X-Custom'
}
request_headers = Headers(raw=[(k.encode(), v.encode()) for k, v in mock_headers.items()])

response = middleware.preflight_response(request_headers)
print(f"Response status: {response.status_code}")
print(f"Response headers: {dict(response.headers)}")

# Test with a header that's in the list
print("\nTesting with bracket character:")
mock_headers['access-control-request-headers'] = '['
request_headers = Headers(raw=[(k.encode(), v.encode()) for k, v in mock_headers.items()])
response = middleware.preflight_response(request_headers)
print(f"Response status: {response.status_code}")
print(f"Response headers: {dict(response.headers)}")

# Test validation logic directly
print("\n" + "="*50)
print("Testing header validation logic:")
print("="*50)

test_headers = ['x-custom', '[', 'Accept', 'unknown-header']
for header in test_headers:
    is_allowed = header.lower() in middleware.allow_headers
    print(f"  Is '{header}' allowed? {is_allowed} (lowercased: '{header.lower()}')")

print(f"\nStored allow_headers: {middleware.allow_headers}")
print(f"Are they sorted? {middleware.allow_headers == sorted(middleware.allow_headers)}")

# Check the preflight headers that get sent
print("\n" + "="*50)
print("Checking preflight headers construction:")
print("="*50)
print(f"Preflight headers: {middleware.preflight_headers}")