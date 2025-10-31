#!/usr/bin/env python3
"""Hypothesis test from bug report"""

from hypothesis import given, strategies as st, settings
import dask.dataframe.io.parquet.core as parquet_core

@given(st.lists(
    st.fixed_dictionaries({
        'columns': st.lists(
            st.fixed_dictionaries({
                'name': st.text(min_size=1, max_size=20),
                'min': st.one_of(st.none(), st.integers(-1000, 1000)),
                'max': st.one_of(st.none(), st.integers(-1000, 1000))
            }),
            min_size=1,
            max_size=5
        )
    }),
    min_size=1,
    max_size=10
))
@settings(max_examples=100)
def test_sorted_columns_divisions_are_sorted(statistics):
    try:
        result = parquet_core.sorted_columns(statistics)
        for item in result:
            assert item['divisions'] == sorted(item['divisions'])
        print(".", end="", flush=True)
    except TypeError as e:
        print(f"\nFound failing case with TypeError: {e}")
        print(f"Statistics: {statistics}")
        raise

if __name__ == "__main__":
    print("Running hypothesis test...")
    try:
        test_sorted_columns_divisions_are_sorted()
        print("\nTest completed!")
    except Exception as e:
        print(f"\nTest failed: {e}")