#!/usr/bin/env python3
"""Hypothesis test that discovers the dask.diagnostics.profile_visualize.unquote bug"""

from hypothesis import given, strategies as st
from dask.diagnostics.profile_visualize import unquote

@given(
    items=st.lists(st.tuples(st.text(), st.integers()), min_size=0, max_size=5)
)
def test_unquote_handles_dict(items):
    expr = (dict, [items])
    result = unquote(expr)
    assert isinstance(result, dict)

# Run the test
if __name__ == "__main__":
    test_unquote_handles_dict()