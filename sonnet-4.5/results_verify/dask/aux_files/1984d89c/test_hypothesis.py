#!/usr/bin/env python3
"""Run hypothesis test from bug report"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages')

from hypothesis import given, assume, strategies as st, settings
from dask.dataframe.dask_expr._repartition import _clean_new_division_boundaries

failures = []

@given(
    boundaries=st.lists(st.integers(min_value=0, max_value=100), min_size=2, max_size=10),
    frame_npartitions=st.integers(min_value=1, max_value=100)
)
@settings(max_examples=1000)
def test_clean_boundaries_invariants(boundaries, frame_npartitions):
    assume(len(boundaries) >= 2)
    assume(all(b >= 0 for b in boundaries))

    result = _clean_new_division_boundaries(boundaries.copy(), frame_npartitions)

    try:
        assert result[0] == 0, f"First boundary should always be 0, got {result[0]}"
        assert result[-1] == frame_npartitions, f"Last boundary should be frame_npartitions ({frame_npartitions}), got {result[-1]}"

        for i in range(len(result) - 1):
            assert result[i] <= result[i+1], f"Boundaries should be non-decreasing at index {i}: {result}"
    except AssertionError as e:
        failures.append((boundaries, frame_npartitions, result, str(e)))
        raise

# Run the test
print("Running hypothesis test...")
try:
    test_clean_boundaries_invariants()
    print("All tests passed!")
except Exception as e:
    print(f"Tests failed. Found {len(failures)} failures.")
    print("\nFirst 5 failures:")
    for i, (boundaries, frame_npartitions, result, error) in enumerate(failures[:5]):
        print(f"\n{i+1}. boundaries={boundaries}, frame_npartitions={frame_npartitions}")
        print(f"   Result: {result}")
        print(f"   Error: {error}")