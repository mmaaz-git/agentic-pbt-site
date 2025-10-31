#!/usr/bin/env python3
"""
Property-based test using Hypothesis to find inconsistencies in is_valid_tag.
This test checks that the function returns the same result regardless of
whether the input is a regular str or an EncodedString.
"""

from hypothesis import given, strategies as st, settings, example
from Cython.Debugger.DebugWriter import is_valid_tag
from Cython.Compiler.StringEncoding import EncodedString


@given(st.text())
@example('.0')
@example('.123')
@example('.999')
@settings(max_examples=1000)
def test_is_valid_tag_consistency(s):
    """
    Test that is_valid_tag returns the same result for both str and EncodedString inputs.
    The function should behave consistently regardless of input type.
    """
    regular_result = is_valid_tag(s)
    encoded_result = is_valid_tag(EncodedString(s))
    assert regular_result == encoded_result, \
        f"Inconsistent: is_valid_tag({s!r}) = {regular_result}, but is_valid_tag(EncodedString({s!r})) = {encoded_result}"


if __name__ == "__main__":
    # Run the test
    test_is_valid_tag_consistency()