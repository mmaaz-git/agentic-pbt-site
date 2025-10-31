#!/usr/bin/env python3
"""Test the reported CORS header case sensitivity bug."""

from hypothesis import given, strategies as st
from starlette.middleware.cors import CORSMiddleware

# First, run the property-based test
@given(
    header=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("Lu", "Ll"), whitelist_characters="-"))
)
def test_cors_headers_case_insensitive_property(header):
    """Test that headers with different capitalizations produce the same allow_headers."""
    middleware_upper = CORSMiddleware(None, allow_headers=[header.upper()])
    middleware_lower = CORSMiddleware(None, allow_headers=[header.lower()])

    assert middleware_upper.allow_headers == middleware_lower.allow_headers, \
        f"Headers differ for '{header}': upper={middleware_upper.allow_headers}, lower={middleware_lower.allow_headers}"

# Run the hypothesis test
print("Running property-based test...")
try:
    test_cors_headers_case_insensitive_property()
    print("Property test passed!")
except AssertionError as e:
    print(f"Property test failed: {e}")
except Exception as e:
    print(f"Error in property test: {e}")

print("\n" + "="*60)
print("Reproducing the specific bug with 'A' vs 'a'...")
print("="*60 + "\n")

# Reproduce the specific bug
middleware_upper = CORSMiddleware(None, allow_headers=['A'])
middleware_lower = CORSMiddleware(None, allow_headers=['a'])

print("With 'A':", middleware_upper.allow_headers)
print("With 'a':", middleware_lower.allow_headers)
print("Equal?", middleware_upper.allow_headers == middleware_lower.allow_headers)

print("\n" + "="*60)
print("Testing with more complex headers...")
print("="*60 + "\n")

# Test with more complex headers
middleware_custom_upper = CORSMiddleware(None, allow_headers=['X-Custom-Header'])
middleware_custom_lower = CORSMiddleware(None, allow_headers=['x-custom-header'])

print("With 'X-Custom-Header':", middleware_custom_upper.allow_headers)
print("With 'x-custom-header':", middleware_custom_lower.allow_headers)
print("Equal?", middleware_custom_upper.allow_headers == middleware_custom_lower.allow_headers)

print("\n" + "="*60)
print("Analyzing the sorting behavior...")
print("="*60 + "\n")

# Let's manually trace through what happens
SAFELISTED_HEADERS = {"Accept", "Accept-Language", "Content-Language", "Content-Type"}

# Case 1: allow_headers=['A']
headers_with_A = SAFELISTED_HEADERS | {'A'}
sorted_with_A = sorted(headers_with_A)
lowercased_A = [h.lower() for h in sorted_with_A]
print("With 'A':")
print("  Before sorting:", headers_with_A)
print("  After sorting:", sorted_with_A)
print("  After lowercasing:", lowercased_A)

# Case 2: allow_headers=['a']
headers_with_a = SAFELISTED_HEADERS | {'a'}
sorted_with_a = sorted(headers_with_a)
lowercased_a = [h.lower() for h in sorted_with_a]
print("\nWith 'a':")
print("  Before sorting:", headers_with_a)
print("  After sorting:", sorted_with_a)
print("  After lowercasing:", lowercased_a)

print("\nKey observation: The sorting happens BEFORE lowercasing, causing different orders!")
print(f"'A' < 'Accept': {('A' < 'Accept')}")
print(f"'a' < 'Accept': {('a' < 'Accept')}")