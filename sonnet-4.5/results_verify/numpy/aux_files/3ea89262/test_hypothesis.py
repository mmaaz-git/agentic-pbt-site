#!/usr/bin/env python3
"""Run the hypothesis test from the bug report"""

from hypothesis import given, strategies as st, settings
from numpy import matrix

@given(st.text(alphabet="0123456789 ,;", min_size=1, max_size=50))
@settings(max_examples=100)
def test_string_parsing_creates_valid_matrices(s):
    try:
        m = matrix(s)
        assert m.ndim == 2, f"Matrix must be 2D, got {m.ndim}D"
        assert all(dim > 0 for dim in m.shape), f"Matrix has zero dimension: {m.shape}"
    except (ValueError, SyntaxError):
        pass

if __name__ == "__main__":
    print("Running hypothesis test...")
    try:
        test_string_parsing_creates_valid_matrices()
        print("Test passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")
    except Exception as e:
        print(f"Found failing case with exception: {e}")