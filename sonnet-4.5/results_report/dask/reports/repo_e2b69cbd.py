from starlette.middleware.cors import CORSMiddleware
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.testclient import TestClient

# Create a simple async app
async def dummy_app(scope, receive, send):
    response = JSONResponse({"message": "Hello"})
    await response(scope, receive, send)

# Configure CORSMiddleware with a header that has trailing whitespace
middleware = CORSMiddleware(
    dummy_app,
    allow_origins=["http://example.com"],
    allow_headers=["X-Custom-Header ", "Content-Type"]  # Note the trailing space in X-Custom-Header
)

# Check what headers are stored internally
print("Stored allow_headers:", middleware.allow_headers)
print()

# Now let's test what happens with a preflight request
app = Starlette()

@app.route("/")
async def homepage(request):
    return JSONResponse({"message": "Hello"})

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://example.com"],
    allow_headers=["X-Custom-Header ", "Content-Type"]  # Note the trailing space
)

client = TestClient(app)

# Test a preflight request with the header (without trailing space)
print("Testing preflight request with 'X-Custom-Header' (no trailing space):")
response = client.options(
    "/",
    headers={
        "Origin": "http://example.com",
        "Access-Control-Request-Method": "GET",
        "Access-Control-Request-Headers": "X-Custom-Header"  # No trailing space
    }
)
print(f"Status Code: {response.status_code}")
print(f"Response: {response.text}")
print()

# Test a preflight request with the header (with trailing space to match config)
print("Testing preflight request with 'X-Custom-Header ' (with trailing space):")
response = client.options(
    "/",
    headers={
        "Origin": "http://example.com",
        "Access-Control-Request-Method": "GET",
        "Access-Control-Request-Headers": "X-Custom-Header "  # With trailing space
    }
)
print(f"Status Code: {response.status_code}")
print(f"Response: {response.text}")