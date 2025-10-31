#!/usr/bin/env python3
"""Property-based test for dask ORC columns mutation bug"""

from hypothesis import given, strategies as st, settings
from dask.dataframe.io.orc.core import _read_orc
from dask.dataframe.io.orc.arrow import ArrowORCEngine


@given(
    index=st.one_of(st.none(), st.text(min_size=1)),
    columns=st.one_of(st.none(), st.lists(st.text(min_size=1), min_size=1, max_size=5))
)
@settings(max_examples=100)
def test_read_orc_does_not_mutate_columns(index, columns):
    """Test that _read_orc does not mutate the input columns list"""
    if columns is not None:
        columns_before = list(columns)

        try:
            _read_orc(
                parts=[],
                engine=ArrowORCEngine,
                fs=None,
                schema={},
                index=index,
                columns=columns
            )
        except:
            pass

        assert columns == columns_before, f"columns was mutated from {columns_before} to {columns}"


if __name__ == "__main__":
    # Run the test
    print("Running property-based test for _read_orc columns mutation...")
    test_read_orc_does_not_mutate_columns()