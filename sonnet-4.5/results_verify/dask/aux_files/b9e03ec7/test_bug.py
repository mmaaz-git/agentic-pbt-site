#!/usr/bin/env python3
"""Test script to reproduce the apply_filters bug"""

import sys
import traceback

# First test: Simple reproduction
try:
    print("Test 1: Simple reproduction of the bug")
    import dask.dataframe.io.parquet.core as parquet_core

    parts = ['part1']
    statistics = [{'columns': []}]
    filters = []

    print(f"  Input parts: {parts}")
    print(f"  Input statistics: {statistics}")
    print(f"  Input filters: {filters}")

    out_parts, out_statistics = parquet_core.apply_filters(parts, statistics, filters)
    print(f"  Success! Output parts: {out_parts}, Output statistics: {out_statistics}")
except Exception as e:
    print(f"  ERROR: {type(e).__name__}: {e}")
    traceback.print_exc()

print()

# Second test: Hypothesis test (simplified version)
try:
    print("Test 2: Hypothesis-based test (simplified)")
    from hypothesis import given, strategies as st, assume
    import dask.dataframe.io.parquet.core as parquet_core

    @given(
        st.lists(st.text(min_size=1), min_size=1, max_size=10),
        st.lists(st.dictionaries(
            st.text(min_size=1),
            st.one_of(st.none(), st.integers(), st.floats(allow_nan=False)),
            min_size=0,
            max_size=10
        ), min_size=1, max_size=10)
    )
    def test_apply_filters_returns_subset(parts, statistics):
        assume(len(parts) == len(statistics))
        for stats in statistics:
            stats['columns'] = []
        filters = []
        out_parts, out_statistics = parquet_core.apply_filters(parts, statistics, filters)
        assert len(out_parts) <= len(parts)

    # Try to run the test
    print("  Running hypothesis test...")
    test_apply_filters_returns_subset()
    print("  Test passed!")

except Exception as e:
    print(f"  ERROR during hypothesis test: {type(e).__name__}: {e}")
    traceback.print_exc()

print()

# Test 3: Check behavior with non-empty filters
try:
    print("Test 3: Check with non-empty filters works")
    import dask.dataframe.io.parquet.core as parquet_core

    parts = ['part1', 'part2']
    statistics = [
        {'columns': [{'name': 'x', 'min': 0, 'max': 10}]},
        {'columns': [{'name': 'x', 'min': 5, 'max': 15}]}
    ]
    filters = [('x', '>', 8)]

    print(f"  Input parts: {parts}")
    print(f"  Input statistics (abbreviated): {len(statistics)} items")
    print(f"  Input filters: {filters}")

    out_parts, out_statistics = parquet_core.apply_filters(parts, statistics, filters)
    print(f"  Success! Filtered to {len(out_parts)} parts")

except Exception as e:
    print(f"  ERROR: {type(e).__name__}: {e}")

print()

# Test 4: Check the exact line that causes the issue
try:
    print("Test 4: Reproduce exact error line")
    filters = []
    print(f"  Filters: {filters}")
    print(f"  Attempting: filters[0] access...")
    result = filters[0]
    print(f"  Should not reach here")
except IndexError as e:
    print(f"  Expected IndexError: {e}")

print()

# Test 5: What the code is trying to do
try:
    print("Test 5: Show what line 556 is doing")

    # Case 1: Empty filters
    filters = []
    print(f"  Case 1 - Empty filters: {filters}")
    try:
        result = filters[0] if filters else None
        print(f"    filters[0] check: {result}")
        # This is what line 556 tries to do:
        conjunction, *disjunction = filters if isinstance(filters[0], list) else [filters]
    except IndexError:
        print(f"    ERROR: Cannot access filters[0] on empty list")

    # Case 2: Simple filter list
    filters = [('x', '>', 5)]
    print(f"\n  Case 2 - Simple filter: {filters}")
    print(f"    filters[0]: {filters[0]}")
    print(f"    isinstance(filters[0], list): {isinstance(filters[0], list)}")
    conjunction, *disjunction = filters if isinstance(filters[0], list) else [filters]
    print(f"    Result: conjunction={conjunction}, disjunction={disjunction}")

    # Case 3: DNF filter list
    filters = [[('x', '>', 5)], [('y', '<', 10)]]
    print(f"\n  Case 3 - DNF filter: {filters}")
    print(f"    filters[0]: {filters[0]}")
    print(f"    isinstance(filters[0], list): {isinstance(filters[0], list)}")
    conjunction, *disjunction = filters if isinstance(filters[0], list) else [filters]
    print(f"    Result: conjunction={conjunction}, disjunction={disjunction}")

except Exception as e:
    print(f"  ERROR: {e}")