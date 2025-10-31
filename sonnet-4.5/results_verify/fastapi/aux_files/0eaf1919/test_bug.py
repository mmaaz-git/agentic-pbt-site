#!/usr/bin/env python3
"""Test the reported CORSMiddleware bug."""

from hypothesis import given, strategies as st, settings
from starlette.middleware.cors import CORSMiddleware

# First, let's run the hypothesis test
@given(st.lists(st.text(min_size=1)))
@settings(max_examples=500)
def test_cors_allow_headers_sorted_and_lowercased(headers):
    middleware = CORSMiddleware(
        app=lambda scope, receive, send: None,
        allow_headers=headers
    )

    stored = middleware.allow_headers

    assert stored == sorted(stored), f"Headers not sorted: {stored} != {sorted(stored)}"

# Try to run the test
print("Running hypothesis test...")
try:
    test_cors_allow_headers_sorted_and_lowercased()
    print("Hypothesis test passed!")
except AssertionError as e:
    print(f"Hypothesis test failed: {e}")

# Now reproduce the specific failing case
print("\n" + "="*50)
print("Reproducing specific failing case: headers=['[']")
print("="*50)

middleware = CORSMiddleware(
    app=lambda scope, receive, send: None,
    allow_headers=['[']
)

print(f"middleware.allow_headers: {middleware.allow_headers}")
print(f"sorted(middleware.allow_headers): {sorted(middleware.allow_headers)}")

try:
    assert middleware.allow_headers == sorted(middleware.allow_headers)
    print("Assertion passed!")
except AssertionError:
    print("Assertion failed: Headers are not in sorted order!")

# Let's also test with other examples that might break
print("\n" + "="*50)
print("Testing with uppercase letters that sort differently")
print("="*50)

test_cases = [
    ['Z', 'a'],  # 'Z' < 'a' in ASCII, but 'z' > 'a' after lowercasing
    ['X-Custom', 'accept'],  # Mixed case custom header
    ['UPPERCASE', 'lowercase'],
    ['['],  # Special character from the bug report
]

for headers in test_cases:
    print(f"\nTesting with headers={headers}")
    middleware = CORSMiddleware(
        app=lambda scope, receive, send: None,
        allow_headers=headers
    )
    print(f"  Input headers: {headers}")
    print(f"  Stored headers: {middleware.allow_headers}")
    print(f"  Sorted stored: {sorted(middleware.allow_headers)}")
    is_sorted = middleware.allow_headers == sorted(middleware.allow_headers)
    print(f"  Is sorted? {is_sorted}")