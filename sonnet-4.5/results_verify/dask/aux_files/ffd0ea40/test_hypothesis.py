#!/usr/bin/env python3
"""Run the hypothesis test from the bug report"""

import numpy as np
from hypothesis import given, strategies as st, assume, settings, example
from dask.dataframe.io.io import sorted_division_locations

@given(
    seq=st.lists(st.integers(min_value=0, max_value=100), min_size=1, max_size=100),
    npartitions=st.integers(min_value=1, max_value=50)
)
@settings(max_examples=100)
@example(seq=[0, 0], npartitions=2)  # The failing case from the bug report
def test_sorted_division_locations_with_npartitions(seq, npartitions):
    assume(npartitions <= len(seq))
    seq = np.array(sorted(seq))

    try:
        divisions, locations = sorted_division_locations(seq, npartitions=npartitions)

        print(f"Input: seq={list(seq)[:10]}{'...' if len(seq) > 10 else ''}, npartitions={npartitions}")
        print(f"  Unique values: {len(set(seq))}, Expected divisions: {npartitions + 1}, Got: {len(divisions)}")

        assert len(divisions) == npartitions + 1, \
            f"divisions must have length npartitions + 1, got {len(divisions)}"
    except AssertionError as e:
        print(f"FAILED: {e}")
        print(f"  seq={list(seq)}, npartitions={npartitions}")
        print(f"  divisions={list(divisions)}")
        raise

if __name__ == "__main__":
    test_sorted_division_locations_with_npartitions()