#!/usr/bin/env python3
"""Hypothesis test for get_path_param_names"""

from hypothesis import given, strategies as st, settings
from fastapi.utils import get_path_param_names

@given(st.text())
@settings(max_examples=100)
def test_get_path_param_names_no_empty_strings(path):
    result = get_path_param_names(path)
    for name in result:
        assert name != '', f"Empty parameter name found for path: {path!r}"

# Run the test
print("Running hypothesis test...")
try:
    test_get_path_param_names_no_empty_strings()
    print("Test passed - no issues found")
except AssertionError as e:
    print(f"Test failed: {e}")

# Test specific failing examples
print("\nTesting specific examples that should fail:")
failing_paths = [
    "/users/{}/posts",
    "{}",
    "/api/{}/data",
    "/{}/{}/{}",
    "/mixed/{id}/{}/end"
]

for path in failing_paths:
    try:
        result = get_path_param_names(path)
        for name in result:
            assert name != '', f"Empty parameter name found for path: {path!r}"
        print(f"✓ Path '{path}' passed (no empty params)")
    except AssertionError as e:
        print(f"✗ Path '{path}' failed: {e}")