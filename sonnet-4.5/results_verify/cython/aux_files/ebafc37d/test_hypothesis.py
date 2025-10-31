#!/usr/bin/env python3
"""Hypothesis test from the bug report"""

from hypothesis import given, strategies as st, settings
from Cython.Build.Dependencies import parse_list

@settings(max_examples=1000)
@given(st.lists(st.text(min_size=0, max_size=20), min_size=0, max_size=10))
def test_parse_list_quoted_bracket_format_no_crash(items):
    s = '[' + ', '.join(f'"{item}"' for item in items) + ']'
    result = parse_list(s)
    # Test should not crash

# Run the test
if __name__ == "__main__":
    print("Running Hypothesis test...")
    try:
        test_parse_list_quoted_bracket_format_no_crash()
        print("Test passed!")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

    # Test the specific failing case
    print("\nTesting specific failing case: items=['']")
    try:
        items = ['']
        s = '[' + ', '.join(f'"{item}"' for item in items) + ']'
        print(f"Input string: {repr(s)}")
        result = parse_list(s)
        print(f"Result: {result}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()