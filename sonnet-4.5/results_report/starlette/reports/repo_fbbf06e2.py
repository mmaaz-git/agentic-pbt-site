from starlette.datastructures import Headers
from starlette.middleware.cors import CORSMiddleware

# Create middleware with a header that has trailing whitespace
middleware = CORSMiddleware(
    app=None,
    allow_origins=["*"],
    allow_headers=["X-Custom "]  # Note the trailing space
)

# Create a preflight request with the header (without whitespace)
request_headers = Headers({
    "origin": "http://example.com",
    "access-control-request-method": "GET",
    "access-control-request-headers": "X-Custom"  # No trailing space
})

# Process the preflight request
response = middleware.preflight_response(request_headers)

# Check the response
print(f"Response status code: {response.status_code}")
print(f"Expected: 200")

if response.status_code != 200:
    print(f"ERROR: Request was rejected!")
    print(f"Response body: {response.body.decode()}")
else:
    print("Success: Request was accepted")

# Demonstrate the issue with the simplest case
print("\n--- Simplest failing case ---")
middleware_minimal = CORSMiddleware(
    app=None,
    allow_origins=["*"],
    allow_headers=["a "]  # Single character header with trailing space
)

request_minimal = Headers({
    "origin": "http://example.com",
    "access-control-request-method": "GET",
    "access-control-request-headers": "a"
})

response_minimal = middleware_minimal.preflight_response(request_minimal)
print(f"Minimal case status: {response_minimal.status_code}")
if response_minimal.status_code != 200:
    print(f"Minimal case error: {response_minimal.body.decode()}")