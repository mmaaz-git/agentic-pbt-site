import numpy as np
import pytest
from hypothesis import given, strategies as st
from pandas.api.indexers import FixedForwardWindowIndexer


@given(st.integers(min_value=1, max_value=100), st.integers(max_value=-1))
def test_fixed_forward_window_negative_size_start_end_invariant(num_values, window_size):
    indexer = FixedForwardWindowIndexer(window_size=window_size)
    start, end = indexer.get_window_bounds(num_values)

    for i in range(len(start)):
        assert start[i] <= end[i], f"Invariant violated: start[{i}]={start[i]} > end[{i}]={end[i]}"


# Direct test with specific failing values
def test_specific_case():
    indexer = FixedForwardWindowIndexer(window_size=-1)
    start, end = indexer.get_window_bounds(num_values=2)

    print(f"start: {start}")
    print(f"end: {end}")

    # Check the invariant
    for i in range(len(start)):
        print(f"  start[{i}]={start[i]}, end[{i}]={end[i]}, valid={start[i] <= end[i]}")

    # Specific check for index 1
    if len(start) > 1:
        print(f"\nstart[1] > end[1]: {start[1]} > {end[1]} = {start[1] > end[1]}")


if __name__ == "__main__":
    print("Running specific failing case:")
    test_specific_case()

    print("\nRunning hypothesis test:")
    try:
        test_fixed_forward_window_negative_size_start_end_invariant(2, -1)
        print("Test passed for num_values=2, window_size=-1")
    except AssertionError as e:
        print(f"Test failed with AssertionError: {e}")