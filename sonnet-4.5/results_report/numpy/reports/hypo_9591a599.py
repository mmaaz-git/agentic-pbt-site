#!/usr/bin/env python3
"""Hypothesis test demonstrating numpy.strings null byte handling bug"""

import numpy as np
import numpy.strings as ns
from hypothesis import given, strategies as st


@given(st.lists(st.just('\x00'), min_size=1))
def test_upper_preserves_null_bytes(strings):
    """Test that numpy.strings.upper preserves null bytes like Python's str.upper()"""
    arr = np.array(strings, dtype=np.str_)
    result = ns.upper(arr)

    for orig, res in zip(strings, result):
        expected = orig.upper()
        assert res == expected, f"Expected {repr(expected)}, got {repr(res)}"


if __name__ == "__main__":
    # Run the test
    test_upper_preserves_null_bytes()