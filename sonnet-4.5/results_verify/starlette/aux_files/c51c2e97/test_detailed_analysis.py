#!/usr/bin/env python3
"""Detailed analysis of the bug"""
from starlette.datastructures import Headers
from starlette.middleware.cors import CORSMiddleware

print("=== Analyzing the processing inconsistency ===\n")

# Looking at lines 58 and 67 of cors.py
allow_headers_input = [' X-Custom-Header ', 'X-Another-Header']

# Step 1: Line 58 - adds safelisted headers and sorts
SAFELISTED_HEADERS = {"Accept", "Accept-Language", "Content-Language", "Content-Type"}
allow_headers_after_line_58 = sorted(SAFELISTED_HEADERS | set(allow_headers_input))
print(f"After line 58 (sorted with safelisted): {allow_headers_after_line_58}")

# Step 2: Line 67 - lowercases but doesn't strip
allow_headers_after_line_67 = [h.lower() for h in allow_headers_after_line_58]
print(f"After line 67 (lowercased): {allow_headers_after_line_67}")

print("\n=== What happens during preflight_response ===\n")

# Line 128: headers are split and lowercased
requested_headers = " X-Custom-Header , X-Another-Header"
headers_after_split = [h.lower() for h in requested_headers.split(",")]
print(f"After split and lower (line 128): {headers_after_split}")

# Line 129: headers are stripped before checking
for header in headers_after_split:
    stripped = header.strip()
    in_allowed = stripped in allow_headers_after_line_67
    print(f"Header: '{header}' -> stripped: '{stripped}' -> in allowed list? {in_allowed}")

print("\n=== Testing various whitespace scenarios ===\n")

test_cases = [
    ("No whitespace", ["X-Custom-Header"], "X-Custom-Header"),
    ("Leading space", [" X-Custom-Header"], " X-Custom-Header"),
    ("Trailing space", ["X-Custom-Header "], "X-Custom-Header "),
    ("Both spaces", [" X-Custom-Header "], " X-Custom-Header "),
    ("Tab character", ["\tX-Custom-Header"], "\tX-Custom-Header"),
    ("Non-breaking space", ["\xa0X-Custom-Header"], "\xa0X-Custom-Header"),
    ("Multiple headers with spaces", ["X-Header-1 ", " X-Header-2"], "X-Header-1 , X-Header-2"),
]

for name, allowed, requested in test_cases:
    print(f"\n{name}:")
    print(f"  Allowed: {allowed}")
    print(f"  Requested: '{requested}'")

    middleware = CORSMiddleware(
        None,
        allow_origins=["https://example.com"],
        allow_methods=["GET"],
        allow_headers=allowed
    )

    headers = Headers({
        "origin": "https://example.com",
        "access-control-request-method": "GET",
        "access-control-request-headers": requested
    })

    response = middleware.preflight_response(headers)
    print(f"  Result: {response.status_code} ({'OK' if response.status_code == 200 else 'FAILED'})")
    print(f"  Stored headers: {[h for h in middleware.allow_headers if 'x-' in h]}")

print("\n=== Testing if headers should have whitespace ===\n")

# According to HTTP specs, header field-names should not have whitespace
# But header values in Access-Control-Request-Headers might have spaces around commas

test_with_comma_spaces = [
    ("Spaces around comma in request", ["X-Header-1", "X-Header-2"], "X-Header-1 , X-Header-2"),
    ("No spaces around comma", ["X-Header-1", "X-Header-2"], "X-Header-1,X-Header-2"),
]

for name, allowed, requested in test_with_comma_spaces:
    print(f"\n{name}:")
    print(f"  Allowed: {allowed}")
    print(f"  Requested: '{requested}'")

    middleware = CORSMiddleware(
        None,
        allow_origins=["https://example.com"],
        allow_methods=["GET"],
        allow_headers=allowed
    )

    headers = Headers({
        "origin": "https://example.com",
        "access-control-request-method": "GET",
        "access-control-request-headers": requested
    })

    response = middleware.preflight_response(headers)
    print(f"  Result: {response.status_code} ({'OK' if response.status_code == 200 else 'FAILED'})")