#!/usr/bin/env python3
"""Test the CORS header case-sensitivity bug"""

from hypothesis import given, strategies as st
from starlette.middleware.cors import CORSMiddleware

def dummy_app(scope, receive, send):
    pass

# First, run the hypothesis test
print("=== Running Hypothesis Test ===")

def test_cors_case_sensitivity(headers):
    lowercased = [h.lower() for h in headers]
    uppercased = [h.upper() for h in headers]

    middleware_lower = CORSMiddleware(dummy_app, allow_headers=lowercased)
    middleware_upper = CORSMiddleware(dummy_app, allow_headers=uppercased)

    assert middleware_lower.allow_headers == middleware_upper.allow_headers, \
        f"Headers differ: lower={middleware_lower.allow_headers}, upper={middleware_upper.allow_headers}"

# Test with the failing input
try:
    test_cors_case_sensitivity(['A'])
    print("Test passed with ['A']")
except AssertionError as e:
    print(f"Test failed with ['A']: {e}")

# Now run the specific reproduction case
print("\n=== Reproduction Case ===")

def app(scope, receive, send):
    pass

m1 = CORSMiddleware(app, allow_headers=['custom-header'])
m2 = CORSMiddleware(app, allow_headers=['CUSTOM-HEADER'])

print("With 'custom-header':", m1.allow_headers)
print("With 'CUSTOM-HEADER':", m2.allow_headers)
print("Are they equal?", m1.allow_headers == m2.allow_headers)

# Let's also test what happens with the preflight headers
print("\n=== Preflight Headers (sent to client) ===")
print("Preflight headers with 'custom-header':", m1.preflight_headers.get("Access-Control-Allow-Headers"))
print("Preflight headers with 'CUSTOM-HEADER':", m2.preflight_headers.get("Access-Control-Allow-Headers"))

# Let's look deeper at the issue
print("\n=== Deeper Analysis ===")
print("SAFELISTED_HEADERS:", sorted(list({'Accept', 'Accept-Language', 'Content-Language', 'Content-Type'})))

# Simulate what happens in line 58
from starlette.middleware.cors import SAFELISTED_HEADERS
headers1 = sorted(SAFELISTED_HEADERS | set(['custom-header']))
headers2 = sorted(SAFELISTED_HEADERS | set(['CUSTOM-HEADER']))
print(f"After line 58 with 'custom-header': {headers1}")
print(f"After line 58 with 'CUSTOM-HEADER': {headers2}")

# Then what happens in line 67
headers1_lower = [h.lower() for h in headers1]
headers2_lower = [h.lower() for h in headers2]
print(f"After line 67 with 'custom-header': {headers1_lower}")
print(f"After line 67 with 'CUSTOM-HEADER': {headers2_lower}")