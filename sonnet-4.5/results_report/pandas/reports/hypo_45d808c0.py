#!/usr/bin/env python3
"""Hypothesis test for the cap_length bug."""

from hypothesis import given, strategies as st, settings
from Cython.Compiler import PyrexTypes


@given(st.text(alphabet=st.characters(min_codepoint=32, max_codepoint=126)),
       st.integers(min_value=10, max_value=200))
@settings(max_examples=1000)
def test_cap_length_respects_max_len(s, max_len):
    result = PyrexTypes.cap_length(s, max_len)
    assert len(result) <= max_len, f"cap_length({s!r}, {max_len}) returned {result!r} with length {len(result)} > {max_len}"


if __name__ == '__main__':
    test_cap_length_respects_max_len()