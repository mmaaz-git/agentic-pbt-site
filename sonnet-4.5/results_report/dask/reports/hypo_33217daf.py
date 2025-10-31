#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages')

from hypothesis import given, assume, strategies as st, settings
from dask.dataframe.dask_expr._repartition import _clean_new_division_boundaries

@given(
    boundaries=st.lists(st.integers(min_value=0, max_value=100), min_size=2, max_size=10),
    frame_npartitions=st.integers(min_value=1, max_value=100)
)
@settings(max_examples=1000)
def test_clean_boundaries_invariants(boundaries, frame_npartitions):
    assume(len(boundaries) >= 2)
    assume(all(b >= 0 for b in boundaries))

    result = _clean_new_division_boundaries(boundaries, frame_npartitions)

    assert result[0] == 0, f"First boundary should always be 0, got {result[0]}"
    assert result[-1] == frame_npartitions, f"Last boundary should be frame_npartitions ({frame_npartitions}), got {result[-1]}"

    for i in range(len(result) - 1):
        assert result[i] <= result[i+1], f"Boundaries should be non-decreasing at index {i}: {result}"

if __name__ == "__main__":
    test_clean_boundaries_invariants()