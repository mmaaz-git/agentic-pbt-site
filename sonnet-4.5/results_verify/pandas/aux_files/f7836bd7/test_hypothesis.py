#!/usr/bin/env python3
"""Property-based test for pandas.io.excel._util._excel2num bug"""

import string
from hypothesis import given, strategies as st, example
from pandas.io.excel._util import _excel2num
import pytest

@given(st.text(min_size=1, max_size=5))
@example(' ')  # Single space
@example('')   # Empty string
@example('\t') # Tab
@example('\n') # Newline
@example('   ') # Multiple spaces
def test_excel2num_invalid_chars_raise_error(text):
    """_excel2num should raise ValueError for invalid characters"""
    if not all(c.isalpha() for c in text):
        with pytest.raises(ValueError, match="Invalid column name"):
            _excel2num(text)

if __name__ == "__main__":
    # Run the test with specific failing examples
    print("Testing specific failing examples from the bug report...")

    failing_inputs = [' ', '', '\t', '\n', '   ']

    for inp in failing_inputs:
        print(f"\nTesting input: {repr(inp)}")
        # Test if the input contains only non-alphabetic characters
        if not all(c.isalpha() for c in inp):
            try:
                # We expect ValueError to be raised
                result = _excel2num(inp)
                print(f"  ✗ Test failed - ValueError was NOT raised")
                print(f"    Function returned: {result}")
            except ValueError as e:
                print(f"  ✓ Test passed (ValueError was raised as expected: {e})")
        else:
            print(f"  Skipping - input contains only alphabetic characters")