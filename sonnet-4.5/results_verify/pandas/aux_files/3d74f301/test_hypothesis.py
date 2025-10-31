#!/usr/bin/env python3
"""Hypothesis test from the bug report"""

from hypothesis import assume, given, settings, strategies as st
import pandas.core.indexers as indexers

@given(
    start=st.one_of(st.integers(), st.none()),
    stop=st.one_of(st.integers(), st.none()),
    step=st.one_of(st.integers(), st.none()),
    n=st.integers(min_value=0, max_value=100)
)
@settings(max_examples=500)
def test_length_of_indexer_matches_actual_slicing(start, stop, step, n):
    assume(step != 0)
    slc = slice(start, stop, step)
    target = list(range(n))

    try:
        expected_len = len(target[slc])
        calculated_len = indexers.length_of_indexer(slc, target)
        if expected_len != calculated_len:
            print(f"FAIL: slice({start}, {stop}, {step}), n={n}")
            print(f"  Expected: {expected_len}, Got: {calculated_len}")
        assert calculated_len == expected_len
    except (ValueError, ZeroDivisionError):
        pass

# Run the test
test_length_of_indexer_matches_actual_slicing()
print("Test completed")