#!/usr/bin/env python3

from hypothesis import given, strategies as st, settings, example
from Cython.Debugger.DebugWriter import is_valid_tag
from Cython.Compiler.StringEncoding import EncodedString

@given(st.text())
@example('.0')
@example('.123')
@example('.999')
@settings(max_examples=1000)
def test_is_valid_tag_consistency(s):
    regular_result = is_valid_tag(s)
    encoded_result = is_valid_tag(EncodedString(s))
    assert regular_result == encoded_result, \
        f"Inconsistent: is_valid_tag({s!r}) = {regular_result}, but is_valid_tag(EncodedString({s!r})) = {encoded_result}"

# Run the test
test_is_valid_tag_consistency()