from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.testclient import TestClient

# Create a simple app with TrustedHostMiddleware
app = Starlette()

# Configure middleware to accept IPv6 loopback address
middleware = TrustedHostMiddleware(app, allowed_hosts=["[::1]"])

@app.route("/")
async def homepage(request):
    return JSONResponse({"message": "Hello, world!"})

# Apply the middleware
app = TrustedHostMiddleware(app, allowed_hosts=["[::1]"])

# Test with a TestClient using IPv6 address
client = TestClient(app)

print("Testing IPv6 host header: [::1]")
print("=" * 50)

# First demonstrate the internal parsing issue
host_header = "[::1]"
extracted_host = host_header.split(":")[0]
print(f"Host header value: {host_header}")
print(f"What split(':')[0] extracts: {extracted_host}")
print(f"Expected extraction: [::1]")
print(f"Does '{extracted_host}' match '[::1]'? {extracted_host == '[::1]'}")
print()

# Now test the actual middleware behavior
print("Testing actual middleware behavior:")
print("-" * 50)

try:
    # This should work if IPv6 is properly supported, but it will fail
    response = client.get("/", headers={"host": "[::1]"})
    print(f"Response status code: {response.status_code}")
    print(f"Response content: {response.text}")
except Exception as e:
    print(f"Request failed with exception: {e}")

print()
print("Testing with IPv6 address with port: [::1]:8080")
print("-" * 50)
host_with_port = "[::1]:8080"
extracted_with_port = host_with_port.split(":")[0]
print(f"Host header value: {host_with_port}")
print(f"What split(':')[0] extracts: {extracted_with_port}")
print(f"Expected extraction: [::1]")

print()
print("Testing with full IPv6 address: [2001:db8::1]")
print("-" * 50)
full_ipv6 = "[2001:db8::1]"
extracted_full = full_ipv6.split(":")[0]
print(f"Host header value: {full_ipv6}")
print(f"What split(':')[0] extracts: {extracted_full}")
print(f"Expected extraction: [2001:db8::1]")

print()
print("For comparison, testing IPv4 with port (works correctly):")
print("-" * 50)
ipv4_with_port = "127.0.0.1:8080"
extracted_ipv4 = ipv4_with_port.split(":")[0]
print(f"Host header value: {ipv4_with_port}")
print(f"What split(':')[0] extracts: {extracted_ipv4}")
print(f"Expected extraction: 127.0.0.1")
print(f"Correctly extracted? {extracted_ipv4 == '127.0.0.1'}")