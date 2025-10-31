#!/usr/bin/env python3
"""Hypothesis-based tests from the bug report"""

from scipy.io.arff._arffread import split_data_line, RelationalAttribute
from hypothesis import given, strategies as st, settings
import sys

@given(st.text())
@settings(max_examples=100)
def test_split_data_line_handles_any_string(line):
    """
    Property: split_data_line should handle any string input without crashing.

    This fails on empty strings due to unchecked indexing.
    """
    try:
        row, dialect = split_data_line(line)
        assert isinstance(row, (list, tuple))
    except ValueError:
        pass

@given(st.text())
@settings(max_examples=100)
def test_relational_parse_data_no_crash(data_str):
    """
    Property: RelationalAttribute.parse_data should not crash.

    This can trigger the bug when data_str ends with newline or is empty,
    because split('\\n') produces empty strings.
    """
    attr = RelationalAttribute("test")
    attr.attributes = []

    try:
        attr.parse_data(data_str)
    except IndexError as e:
        if "string index out of range" in str(e):
            raise AssertionError(f"IndexError on split_data_line with data: {repr(data_str)}") from e
    except Exception:
        pass

if __name__ == "__main__":
    print("Testing test_split_data_line_handles_any_string...")
    try:
        test_split_data_line_handles_any_string()
        print("PASSED")
    except Exception as e:
        print(f"FAILED: {e}")
        sys.exit(1)

    print("\nTesting test_relational_parse_data_no_crash...")
    try:
        test_relational_parse_data_no_crash()
        print("PASSED")
    except Exception as e:
        print(f"FAILED: {e}")
        sys.exit(1)