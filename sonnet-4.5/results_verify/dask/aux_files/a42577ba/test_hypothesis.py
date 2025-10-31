#!/usr/bin/env python3
"""Run the hypothesis test from the bug report"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, example
from dask.dataframe.io.parquet.core import sorted_columns

@given(
    st.lists(
        st.dictionaries(
            st.just("columns"),
            st.lists(
                st.fixed_dictionaries({
                    "name": st.text(min_size=1, max_size=10),
                    "min": st.one_of(st.none(), st.integers(), st.floats(allow_nan=False, allow_infinity=False)),
                    "max": st.one_of(st.none(), st.integers(), st.floats(allow_nan=False, allow_infinity=False)),
                }),
                min_size=1,
                max_size=5
            ),
            min_size=1,
            max_size=1
        ),
        min_size=1,
        max_size=10
    )
)
@example([{'columns': [{'name': '0', 'min': 0, 'max': None}]}])  # The failing case from the bug report
def test_sorted_columns_divisions_are_sorted(statistics):
    try:
        result = sorted_columns(statistics)
        for item in result:
            divisions = item["divisions"]
            # This assertion checks if divisions are sorted
            assert divisions == sorted(divisions), f"Divisions not sorted: {divisions}"
    except TypeError as e:
        # The bug report says this crashes with TypeError
        print(f"TypeError encountered: {e}")
        print(f"Input that caused error: {statistics}")
        raise

if __name__ == "__main__":
    # Run the test
    print("Running hypothesis test...")
    test_sorted_columns_divisions_are_sorted()
    print("Test completed!")