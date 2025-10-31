#!/usr/bin/env python3
"""Run the hypothesis test from the bug report"""

from hypothesis import given, strategies as st, settings
import xarray.core.formatting as fmt

@given(st.integers(min_value=1, max_value=100))
@settings(max_examples=1000)
def test_pretty_print_length(numchars):
    obj = "test"
    result = fmt.pretty_print(obj, numchars)
    assert len(result) == numchars, f"Expected length {numchars}, got {len(result)} for result '{result}'"

# Run the test
print("Running hypothesis test...")
try:
    test_pretty_print_length()
    print("All tests passed!")
except AssertionError as e:
    print(f"Test failed: {e}")
except Exception as e:
    print(f"Test failed with exception: {e}")