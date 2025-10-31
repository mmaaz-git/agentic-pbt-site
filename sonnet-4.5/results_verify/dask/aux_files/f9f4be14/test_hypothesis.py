#!/usr/bin/env python3
"""Hypothesis test to verify the mutation bug"""

from hypothesis import given, settings, strategies as st, example
import pyarrow as pa
from dask.dataframe.io.orc.arrow import ArrowORCEngine
from dask.dataframe.io.orc.core import _read_orc


@given(
    st.lists(st.text(min_size=1, max_size=10, alphabet=st.characters(min_codepoint=97, max_codepoint=122)), min_size=1, max_size=5, unique=True),
    st.text(min_size=1, max_size=10, alphabet=st.characters(min_codepoint=97, max_codepoint=122))
)
@settings(max_examples=100)
@example(['a'], 'a')  # The specific failing example from the report
def test_read_orc_does_not_mutate_columns(column_names, index_name):
    columns_list = list(column_names)
    original_columns = columns_list.copy()

    parts = [("dummy_path", [0])]
    engine = ArrowORCEngine
    fs = None
    schema = {col: pa.int64() for col in column_names}

    try:
        _read_orc(parts, engine=engine, fs=fs, schema=schema, index=index_name, columns=columns_list)
    except:
        pass

    assert columns_list == original_columns, f"columns list was mutated: {original_columns} -> {columns_list}"

if __name__ == "__main__":
    # Run the test
    test_read_orc_does_not_mutate_columns()