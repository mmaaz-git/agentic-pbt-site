from starlette.datastructures import Headers
from starlette.middleware.cors import CORSMiddleware

# Demonstrate the bug with a trailing space in the allowed origin
middleware = CORSMiddleware(
    app=None,
    allow_origins=["http://example.com "]  # Note: trailing space
)

request_headers = Headers({
    "origin": "http://example.com",  # No trailing space
    "access-control-request-method": "GET"
})

response = middleware.preflight_response(request_headers)

print(f"Configuration: allow_origins=['http://example.com '] (with trailing space)")
print(f"Request origin: 'http://example.com' (no space)")
print(f"Response status: {response.status_code}")
print(f"Expected: 200 (origin should be allowed)")
print(f"Actual: {response.status_code}")

if response.status_code != 200:
    print(f"\nERROR: Valid request was rejected!")
    print(f"Response body: {response.body.decode()}")