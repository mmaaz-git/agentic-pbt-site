#!/usr/bin/env python3
"""
Property-based test for ARGSORT_DEFAULTS duplicate assignment bug.
"""

from hypothesis import given, strategies as st
from pandas.compat.numpy.function import ARGSORT_DEFAULTS


@given(st.just(None))
def test_argsort_defaults_duplicate_assignment(expected):
    """
    Test that ARGSORT_DEFAULTS['kind'] has the value None.

    This test reveals that line 138 in function.py which sets
    'kind' to 'quicksort' is dead code, as it's immediately
    overwritten by line 140 which sets it to None.
    """
    actual = ARGSORT_DEFAULTS['kind']
    assert actual == expected, f"Expected {expected} but got {actual}"

    # Additional assertion to demonstrate the dead code
    # Line 138 sets 'quicksort', but it never takes effect
    assert actual != 'quicksort', "Line 138's assignment should be overwritten"


if __name__ == "__main__":
    # Run the hypothesis test
    test_argsort_defaults_duplicate_assignment()
    print("Hypothesis test passed: ARGSORT_DEFAULTS['kind'] is None")
    print("This confirms that line 138 (setting 'kind' to 'quicksort') is dead code")