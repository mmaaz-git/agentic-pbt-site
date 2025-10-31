#!/usr/bin/env python3
"""Hypothesis-based property test for the unquote function."""

from hypothesis import given, strategies as st
from dask.diagnostics.profile_visualize import unquote
from dask.core import istask

@given(st.lists(st.tuples(st.text(min_size=1), st.integers()), max_size=5))
def test_unquote_handles_dict_task(items):
    task = (dict, [items])
    result = unquote(task)
    if istask(task) and items:
        assert isinstance(result, dict)
        assert result == dict(items)
    else:
        assert result == task

# Run the test
if __name__ == "__main__":
    test_unquote_handles_dict_task()