#!/usr/bin/env python3
"""Hypothesis property test for chars_to_ranges function."""

from hypothesis import given, strategies as st
from Cython.Plex.Regexps import chars_to_ranges

@given(st.text())
def test_chars_to_ranges_preserves_all_characters(s):
    ranges = chars_to_ranges(s)

    assert len(ranges) % 2 == 0

    reconstructed_chars = set()
    for i in range(0, len(ranges), 2):
        code1, code2 = ranges[i], ranges[i + 1]
        for code in range(code1, code2):
            reconstructed_chars.add(chr(code))

    assert reconstructed_chars == set(s)

if __name__ == "__main__":
    # Run the property test
    test_chars_to_ranges_preserves_all_characters()