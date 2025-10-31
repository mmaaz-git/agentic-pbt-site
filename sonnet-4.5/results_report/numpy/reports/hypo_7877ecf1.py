#!/usr/bin/env python3
"""
Property-based test for numpy.char.mod tuple handling bug.
This test demonstrates that numpy.char.mod fails to handle tuple arguments
for format strings with multiple placeholders.
"""

import numpy as np
import numpy.char as char
from hypothesis import given, strategies as st

@given(st.integers(min_value=0, max_value=100))
def test_mod_with_multiple_formats(value):
    """Test that numpy.char.mod handles tuples for multiple format specifiers."""
    # Create a format string with two format specifiers
    format_string = 'value: %d, hex: %x'
    arr = np.array([format_string], dtype=str)

    # This should work the same as Python's % operator
    expected = format_string % (value, value)

    # But numpy.char.mod fails with a tuple
    result = char.mod(arr, (value, value))

    assert result[0] == expected, f"Expected '{expected}' but got '{result[0]}'"

if __name__ == "__main__":
    # Run the test with a simple value to demonstrate the failure
    print("Testing numpy.char.mod with tuple for multiple format specifiers...")
    print("\nRunning property-based test with value=0:")
    format_string = 'value: %d, hex: %x'
    print(f"Python's % operator: '{format_string}' % (0, 0) = '{format_string % (0, 0)}'")
    try:
        arr = np.array([format_string], dtype=str)
        result = char.mod(arr, (0, 0))
        print(f"numpy.char.mod: {result}")
    except Exception as e:
        print(f"numpy.char.mod raised: {type(e).__name__}: {e}")

    # Also run the hypothesis test
    print("\nRunning hypothesis test:")
    try:
        test_mod_with_multiple_formats()
        print("Test passed!")
    except Exception as e:
        print(f"Test failed: {type(e).__name__}: {e}")