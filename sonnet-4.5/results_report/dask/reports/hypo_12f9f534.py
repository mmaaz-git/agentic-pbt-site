#!/usr/bin/env python3
"""
Hypothesis property-based test that reveals the dask.sizeof dict non-determinism bug.
Tests that sizeof for dictionaries should follow a predictable formula.
"""
from hypothesis import given, settings, strategies as st
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages')
from dask.sizeof import sizeof

@given(st.dictionaries(st.text(), st.integers()))
@settings(max_examples=500)
def test_sizeof_dict_formula(d):
    expected = (
        sys.getsizeof(d)
        + sizeof(list(d.keys()))
        + sizeof(list(d.values()))
        - 2 * sizeof(list())
    )
    assert sizeof(d) == expected

if __name__ == "__main__":
    test_sizeof_dict_formula()