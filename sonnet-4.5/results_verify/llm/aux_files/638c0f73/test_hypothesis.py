#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
from llm.default_plugins.openai_models import not_nulls

@given(st.dictionaries(st.text(), st.one_of(st.none(), st.integers(), st.text())))
def test_not_nulls_removes_none_values(data):
    result = not_nulls(data)
    for key, value in result.items():
        assert value is not None

print("Running hypothesis test...")
try:
    test_not_nulls_removes_none_values()
    print("Test passed!")
except Exception as e:
    print(f"Test failed: {type(e).__name__}: {e}")