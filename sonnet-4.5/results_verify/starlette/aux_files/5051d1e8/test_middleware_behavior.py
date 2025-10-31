#!/usr/bin/env python3
"""Test the actual GZipMiddleware behavior with different headers"""

import asyncio
from starlette.middleware.gzip import GZipMiddleware
from starlette.applications import Starlette
from starlette.responses import Response
from starlette.testclient import TestClient

app = Starlette()

@app.route("/")
def homepage(request):
    return Response("Hello, world!" * 100, media_type="text/plain")

# Add GZipMiddleware
app.add_middleware(GZipMiddleware, minimum_size=10)

# Test with TestClient
client = TestClient(app)

test_cases = [
    # (header_value, expected_compression, description)
    ("gzip", True, "lowercase 'gzip' should work"),
    ("GZIP", False, "uppercase 'GZIP' should work per RFC but doesn't"),
    ("GZip", False, "mixed case 'GZip' should work per RFC but doesn't"),
    ("deflate, gzip", True, "multiple encodings with gzip should work"),
    ("deflate, GZIP", False, "multiple encodings with uppercase GZIP doesn't work"),
    ("not-gzip", True, "substring 'not-gzip' incorrectly matches"),
    ("gzip-not", True, "substring 'gzip-not' incorrectly matches"),
    ("gzip, deflate", True, "gzip first in list works"),
    ("gzip;q=1.0", True, "gzip with quality value works"),
    ("GZIP;q=1.0", False, "uppercase GZIP with quality value doesn't work"),
]

print("Testing GZipMiddleware behavior with different Accept-Encoding headers:\n")

for header_value, expected_compression, description in test_cases:
    response = client.get("/", headers={"Accept-Encoding": header_value})
    is_compressed = "content-encoding" in response.headers and response.headers["content-encoding"] == "gzip"

    status = "✓" if is_compressed == expected_compression else "✗"
    print(f"{status} Accept-Encoding: '{header_value}'")
    print(f"  Expected compression: {expected_compression}, Got: {is_compressed}")
    print(f"  Description: {description}")
    print()