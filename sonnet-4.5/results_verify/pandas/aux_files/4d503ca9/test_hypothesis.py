#!/usr/bin/env python3
"""Hypothesis test to reproduce the _split_line bug."""

from hypothesis import given, strategies as st, settings
from pandas.io.sas.sas_xport import _split_line


@given(
    st.lists(
        st.tuples(
            st.text(alphabet=st.characters(blacklist_characters=['\x00']), min_size=1, max_size=10),
            st.integers(min_value=1, max_value=20)
        ),
        min_size=1,
        max_size=10
    )
)
@settings(max_examples=1000)
def test_split_line_key_invariant(parts):
    total_length = sum(length for _, length in parts)
    s = 'x' * total_length

    try:
        result = _split_line(s, parts)
        expected_keys = {name for name, _ in parts if name != '_'}
        assert set(result.keys()) == expected_keys
        print(f"Pass: parts={parts[:2]}...")
    except KeyError as e:
        print(f"KeyError on parts without '_': {parts}")
        raise

# Run the test
test_split_line_key_invariant()