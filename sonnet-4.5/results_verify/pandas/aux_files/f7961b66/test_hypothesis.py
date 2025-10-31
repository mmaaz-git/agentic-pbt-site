#!/usr/bin/env python3
"""Run the hypothesis test from the bug report"""

from hypothesis import given, settings, strategies as st, example
import numpy as np
from pandas.core.indexers import length_of_indexer

@given(
    start=st.integers(min_value=-100, max_value=100) | st.none(),
    stop=st.integers(min_value=-100, max_value=100) | st.none(),
    step=st.integers(min_value=-100, max_value=100).filter(lambda x: x != 0) | st.none(),
    target_len=st.integers(min_value=0, max_value=200)
)
@settings(max_examples=500)
@example(start=1, stop=None, step=None, target_len=0)  # The failing case from the bug report
def test_length_of_indexer_slice(start, stop, step, target_len):
    slc = slice(start, stop, step)
    target = np.arange(target_len)

    computed_length = length_of_indexer(slc, target)
    actual_length = len(target[slc])

    assert computed_length == actual_length, f"Mismatch for slice({start}, {stop}, {step}) on target of length {target_len}: computed={computed_length}, actual={actual_length}"

if __name__ == "__main__":
    print("Running hypothesis test...")
    try:
        test_length_of_indexer_slice()
        print("All tests passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")
    except Exception as e:
        print(f"Error during testing: {e}")