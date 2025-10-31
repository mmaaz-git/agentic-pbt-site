#!/usr/bin/env python3
"""Hypothesis test for scipy.io.arff.split_data_line function."""

from hypothesis import given, strategies as st
from scipy.io.arff._arffread import split_data_line

@given(st.text())
def test_split_data_line_handles_all_strings(line):
    """Test that split_data_line can handle any string input without crashing."""
    try:
        result, dialect = split_data_line(line)
        assert isinstance(result, list)
        print(f"✓ Passed for line: {repr(line)[:50]}")
    except ValueError:
        # ValueError is an expected exception for malformed data
        print(f"✓ ValueError (expected) for line: {repr(line)[:50]}")
    except IndexError as e:
        # IndexError should not happen
        print(f"✗ FAILED with IndexError for line: {repr(line)}")
        print(f"  Error: {e}")
        raise

if __name__ == "__main__":
    print("Running Hypothesis test on split_data_line function...")
    print("-" * 60)
    test_split_data_line_handles_all_strings()
    print("-" * 60)
    print("Test completed successfully!")