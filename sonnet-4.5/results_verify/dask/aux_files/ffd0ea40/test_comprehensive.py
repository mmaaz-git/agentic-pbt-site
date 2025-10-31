#!/usr/bin/env python3
"""Comprehensive testing of sorted_division_locations behavior"""

import numpy as np
from dask.dataframe.io.io import sorted_division_locations

def test_case(name, seq, npartitions=None, chunksize=None):
    print(f"\n=== {name} ===")
    print(f"Input: seq={list(seq) if hasattr(seq, '__len__') and len(seq) <= 10 else str(seq)[:50]+'...'}")
    if npartitions:
        print(f"       npartitions={npartitions}")
    if chunksize:
        print(f"       chunksize={chunksize}")

    try:
        divisions, locations = sorted_division_locations(seq, npartitions=npartitions, chunksize=chunksize)
        print(f"Output: divisions={list(divisions)}")
        print(f"        locations={list(locations)}")

        # Check invariants
        print("\nInvariants:")
        print(f"  len(divisions) = {len(divisions)}")
        print(f"  len(locations) = {len(locations)}")
        if npartitions:
            expected_len = npartitions + 1
            print(f"  Expected len(divisions) = {expected_len} (npartitions + 1)")
            if len(divisions) != expected_len:
                print(f"  *** MISMATCH: Got {len(divisions)} divisions instead of {expected_len} ***")
        print(f"  len(unique values in seq) = {len(set(seq))}")
        print(f"  divisions[0] == seq[0]: {divisions[0] == seq[0]}")
        print(f"  divisions[-1] == seq[-1]: {divisions[-1] == seq[-1]}")
        print(f"  locations[0] == 0: {locations[0] == 0}")
        print(f"  locations[-1] == len(seq): {locations[-1] == len(seq)}")

    except Exception as e:
        print(f"ERROR: {e}")

# Test cases from the bug report
test_case("Bug Report Case 1: All duplicates", np.array([0, 0]), npartitions=2)
test_case("Bug Report Case 2: More duplicates", np.array([5, 5, 5, 5]), npartitions=3)

# Test with various duplicate patterns
test_case("Mixed with duplicates", np.array([1, 1, 2, 2, 3, 3]), npartitions=4)
test_case("Mixed with duplicates (fewer partitions)", np.array([1, 1, 2, 2, 3, 3]), npartitions=2)

# Test when unique values < npartitions
test_case("3 unique values, 5 partitions requested", np.array([1, 1, 1, 2, 2, 2, 3, 3, 3]), npartitions=5)

# Test when unique values == npartitions
test_case("3 unique values, 3 partitions", np.array([1, 1, 2, 2, 3, 3]), npartitions=3)

# Test with no duplicates
test_case("No duplicates", np.array([1, 2, 3, 4, 5]), npartitions=2)
test_case("No duplicates (more partitions)", np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), npartitions=5)

# Edge cases
test_case("Single element", np.array([42]), npartitions=1)
test_case("Single element, 2 partitions requested", np.array([42]), npartitions=2)

# Test with chunksize instead
test_case("Duplicates with chunksize", np.array([1, 1, 1, 2, 2, 2, 3, 3, 3]), chunksize=3)
test_case("All duplicates with chunksize", np.array([5, 5, 5, 5]), chunksize=2)