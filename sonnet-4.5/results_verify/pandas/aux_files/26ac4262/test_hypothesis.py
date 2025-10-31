#!/usr/bin/env python3
"""Property-based test for pandas.io.sas.sas_xport._split_line"""

from hypothesis import given, strategies as st, settings
from pandas.io.sas.sas_xport import _split_line
import pytest


@given(
    parts=st.lists(
        st.tuples(
            st.text(
                alphabet=st.characters(blacklist_characters=["_"]),
                min_size=1,
                max_size=10
            ),
            st.integers(min_value=1, max_value=20)
        ),
        min_size=1,
        max_size=5
    )
)
@settings(max_examples=10)
def test_split_line_requires_underscore(parts):
    total_length = sum(length for _, length in parts)
    s = "x" * total_length

    try:
        result = _split_line(s, parts)
        pytest.fail(f"Expected KeyError when no '_' field in parts: {parts}")
    except KeyError as e:
        assert "'_'" in str(e) or "_" in str(e)
        print(f"✓ KeyError raised as expected for parts: {parts}")

if __name__ == "__main__":
    # Run the test directly
    test_split_line_requires_underscore()
    print("\nAll tests passed - function always raises KeyError without '_' field")