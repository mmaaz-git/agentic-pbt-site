#!/usr/bin/env python3
"""Hypothesis test for not_nulls function"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
from llm.default_plugins.openai_models import not_nulls

@given(st.dictionaries(st.text(), st.one_of(st.none(), st.integers(), st.text())))
def test_not_nulls_with_dict(data):
    """Property: not_nulls should filter out None values from a dict"""
    result = not_nulls(data)
    assert all(v is not None for v in result.values())

if __name__ == "__main__":
    # Run the test
    test_not_nulls_with_dict()