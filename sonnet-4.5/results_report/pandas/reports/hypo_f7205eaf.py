#!/usr/bin/env python3
"""
Hypothesis property-based test for pandas slice_replace bug
"""
import pandas as pd
from hypothesis import given, strategies as st, settings


@given(
    st.lists(st.text(min_size=1, max_size=20), min_size=1, max_size=10),
    st.integers(min_value=0, max_value=10),
    st.integers(min_value=0, max_value=10),
    st.text(max_size=10)
)
@settings(max_examples=500)
def test_slice_replace_matches_python(strings, start, stop, repl):
    s = pd.Series(strings)
    pandas_result = s.str.slice_replace(start, stop, repl)

    for i in range(len(s)):
        if isinstance(s.iloc[i], str):
            original = s.iloc[i]
            expected = original[:start] + repl + original[stop:]
            assert pandas_result.iloc[i] == expected


if __name__ == "__main__":
    # Run the test
    test_slice_replace_matches_python()