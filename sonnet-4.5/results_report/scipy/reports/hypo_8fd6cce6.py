#!/usr/bin/env python3
"""
Hypothesis property-based test for scipy.io.arff split_data_line function.
This test verifies that the function can handle all string inputs without crashing.
"""

from hypothesis import given, strategies as st
from scipy.io.arff._arffread import split_data_line

@given(st.text())
def test_split_data_line_handles_all_strings(line):
    """Property test: split_data_line should handle any string input without crashing."""
    result, dialect = split_data_line(line)
    assert isinstance(result, list)

if __name__ == "__main__":
    # Run the test to find a failing example
    try:
        test_split_data_line_handles_all_strings()
        print("All tests passed!")
    except Exception as e:
        print(f"Test failed with exception: {e}")