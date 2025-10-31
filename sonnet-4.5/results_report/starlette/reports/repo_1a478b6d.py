from starlette.datastructures import Headers
from starlette.middleware.cors import CORSMiddleware

# Test case: method with trailing space in config
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

print(f"Configuration: allow_methods=['GET '] (with trailing space)")
print(f"Request: access-control-request-method='GET' (no space)")
print(f"Response status code: {response.status_code}")
print(f"Expected: 200, Got: {response.status_code}")

if response.status_code != 200:
    print("\nERROR: Valid CORS preflight request was rejected!")
    print("The middleware failed to match 'GET' request with 'GET ' in allow_methods")