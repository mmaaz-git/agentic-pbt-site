#!/usr/bin/env python3
"""Run the hypothesis test from the bug report"""

from hypothesis import given, strategies as st, settings
from dask.dataframe.io.io import sorted_division_locations

@given(
    seq=st.lists(st.integers(min_value=-1000, max_value=1000), min_size=1, max_size=100),
    chunksize=st.integers(min_value=1, max_value=20)
)
@settings(max_examples=10)  # Reduced for quick check - will fail on first example
def test_sorted_division_locations_accepts_lists(seq, chunksize):
    seq_sorted = sorted(seq)
    try:
        divisions, locations = sorted_division_locations(seq_sorted, chunksize=chunksize)
        assert divisions[0] == seq_sorted[0]
        assert divisions[-1] == seq_sorted[-1]
        print(f"Test passed for seq={seq_sorted[:5]}... (len={len(seq_sorted)}), chunksize={chunksize}")
    except TypeError as e:
        if "No dispatch for <class 'list'>" in str(e):
            print(f"Failed with expected error for seq={seq_sorted[:5]}... (len={len(seq_sorted)}), chunksize={chunksize}")
            raise
        else:
            raise

# Run the test
test_sorted_division_locations_accepts_lists()