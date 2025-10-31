#!/usr/bin/env python3
"""Test script to reproduce the GZipMiddleware case-sensitive bug"""

from hypothesis import given, settings, strategies as st

# Test 1: Hypothesis test
@given(
    case_variant=st.sampled_from(["gzip", "GZIP", "Gzip", "GZip", "gZip", "GzIp"]),
)
@settings(max_examples=50)
def test_gzip_case_insensitive(case_variant):
    current_match = "gzip" in case_variant
    expected_match = True

    assert current_match == expected_match, \
        f"Bug: Accept-Encoding '{case_variant}' should match 'gzip' (HTTP is case-insensitive), but got {current_match}"

# Test 2: Simple reproduction test
def reproduce_bug():
    from starlette.middleware.gzip import GZipMiddleware

    accept_encoding_upper = "GZIP"
    accept_encoding_substring = "not-gzip"

    current_check_upper = "gzip" in accept_encoding_upper
    current_check_substring = "gzip" in accept_encoding_substring

    print(f"'gzip' in 'GZIP': {current_check_upper}")
    print(f"'gzip' in 'not-gzip': {current_check_substring}")

    return current_check_upper, current_check_substring

if __name__ == "__main__":
    print("=== Running Hypothesis test ===")
    try:
        test_gzip_case_insensitive()
        print("Hypothesis test passed (all case variants matched)")
    except AssertionError as e:
        print(f"Hypothesis test failed: {e}")

    print("\n=== Running simple reproduction ===")
    upper_result, substring_result = reproduce_bug()

    print("\n=== Analysis ===")
    print(f"1. Case-sensitive issue: 'GZIP' should match but got {upper_result}")
    print(f"2. Substring issue: 'not-gzip' should NOT match but got {substring_result}")