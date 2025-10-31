#!/usr/bin/env python3
"""Run the Hypothesis test from the bug report"""

from hypothesis import given, strategies as st
from Cython.Utility import pylong_join, _pylong_join

@given(st.integers(min_value=0, max_value=50))
def test_consistency_between_implementations(count):
    public_result = pylong_join(count)
    private_result = _pylong_join(count)
    assert public_result == private_result, f"For count={count}: pylong_join returned {repr(public_result[:50])}... but _pylong_join returned {repr(private_result[:50])}..."

if __name__ == "__main__":
    try:
        test_consistency_between_implementations()
        print("All tests passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")
        print("\nRunning minimal failing case...")
        # Find the minimal failing example
        for i in range(51):
            try:
                test_consistency_between_implementations(i)
            except AssertionError:
                print(f"First failure at count={i}")
                break