#!/usr/bin/env python3
"""
Demonstrating the GZipMiddleware case-sensitivity and substring matching bugs.
"""

# Direct demonstration of the bug in the logic
print("=== Direct Logic Bug Demonstration ===")
print("Current implementation uses: 'gzip' in headers.get('Accept-Encoding', '')")
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
from starlette.middleware.gzip import GZipMiddleware
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.testclient import TestClient
import gzip

app = Starlette()

@app.route("/")
def homepage(request):
    # Return a response large enough to trigger compression (>500 bytes)
    return PlainTextResponse("X" * 1000)

# Wrap app with GZipMiddleware
app = GZipMiddleware(app, minimum_size=100)

# Create test client
client = TestClient(app)

test_cases = [
    ("gzip", "lowercase - should compress"),
    ("GZIP", "uppercase - should compress but doesn't"),
    ("GZip", "mixed case - should compress but doesn't"),
    ("not-gzip", "substring 'not-gzip' - shouldn't compress but does"),
    ("msgzip", "substring 'msgzip' - shouldn't compress but does"),
    ("deflate", "different encoding - shouldn't compress"),
]

print("Testing different Accept-Encoding headers:")
for encoding, description in test_cases:
    response = client.get("/", headers={"Accept-Encoding": encoding})
    is_compressed = response.headers.get("content-encoding") == "gzip"

    # Try to decompress to verify
    if is_compressed:
        try:
            original_content = gzip.decompress(response.content)
            actual_compression = True
        except:
            actual_compression = False
    else:
        actual_compression = False

    print(f"  {encoding:15} ({description:45}): {'COMPRESSED' if actual_compression else 'NOT compressed'}")

print()
print("=== Bug Summary ===")
print("1. Case-sensitive check violates RFC 7231 (content-coding values are case-insensitive)")
print("2. Substring matching can cause false positives (e.g., 'not-gzip' matches)")
print("3. Valid headers like 'GZIP' or 'GZip' fail to trigger compression")