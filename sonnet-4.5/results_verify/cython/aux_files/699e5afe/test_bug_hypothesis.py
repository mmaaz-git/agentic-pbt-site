#!/usr/bin/env python3
"""Test the bug using the property-based test from the report"""

from hypothesis import given, strategies as st
from Cython.Debugger.DebugWriter import is_valid_tag
from Cython.Compiler.StringEncoding import EncodedString

@given(st.integers(min_value=0, max_value=1000000))
def test_is_valid_tag_rejects_dot_decimal_strings(num):
    name = f".{num}"
    result = is_valid_tag(name)
    assert result == False, f"is_valid_tag('{name}') should return False but returned {result}"

# Run the test
if __name__ == "__main__":
    test_is_valid_tag_rejects_dot_decimal_strings()