#!/usr/bin/env python3
"""Hypothesis test for pandas ujson round-trip property"""

from hypothesis import given, strategies as st, settings
from pandas.io.json import ujson_dumps, ujson_loads
import pandas as pd

@given(
    st.one_of(
        st.integers(),
        st.floats(allow_nan=False, allow_infinity=False),
        st.text(),
        st.booleans(),
        st.none()
    )
)
@settings(max_examples=100, verbosity=2)
def test_ujson_roundtrip(data):
    """Test that ujson_dumps and ujson_loads preserve data accurately"""
    json_str = ujson_dumps(data)
    result = ujson_loads(json_str)
    assert result == data or (pd.isna(result) and pd.isna(data))

if __name__ == "__main__":
    test_ujson_roundtrip()